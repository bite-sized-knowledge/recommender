"""
recommender 배치 파이프라인 — Phase 1+2 (+ Phase 1.6 device bandit / 1.8 stage 병렬).

Stage:
  1. global_ranking.build_pool                : recommendation_global atomic swap (critical, 직렬)
  병렬 그룹 (서로 독립 — 다른 테이블 write):
    2.  engagement_aggregator                  : user_events → user_article_engagement
    3.  bandit_reconcile.reconcile             : member_category_bandit
    3b. bandit_reconcile.reconcile_devices     : device_category_bandit
    4.  user_vector.build_profiles             : Qdrant user_profile (Phase 2)
  5. metric_rollup.rollup                     : 위 4 stage 모두 끝난 후 일별 KPI

각 stage 시간/payload 는 recommender_run_metric 에 저장. 실패는 graceful skip.
SQLAlchemy engine 의 connection pool 이 동시 4 conn 처리 가능 (default 5+ overflow).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from common.db import Connection
from data.pipeline.bandit_reconcile import (
    reconcile as bandit_reconcile,
    reconcile_devices as bandit_reconcile_devices,
)
from data.pipeline.engagement_aggregator import aggregate_engagement
from data.pipeline.global_ranking import build_pool as build_global_pool
from data.pipeline.metric_rollup import rollup as metric_rollup
from data.pipeline.user_vector import build_profiles as build_user_profiles
from utils.config_loader import load_config
from utils.logger import get_logger
from utils.metric_sink import MetricSink

logger = get_logger("Recommender")


def _run_stage(sink: MetricSink, name: str, fn, *args, **kwargs):
    t0 = time.time()
    try:
        payload = fn(*args, **kwargs)
    except Exception as e:
        logger.exception(f"{name} 실패: {e}")
        payload = {"error_class": type(e).__name__, "error": str(e)}
        sink.record(name, int((time.time() - t0) * 1000), payload)
        return False, payload
    sink.record(name, int((time.time() - t0) * 1000), payload)
    return True, payload


def run_pipeline() -> None:
    pipeline_start = time.time()
    conn = None
    sink = None
    overall = {}

    try:
        logger.info("=== Connection & Config loading... ===")
        conn = Connection()
        config = load_config()
        sink = MetricSink(conn)
        logger.info(f"=== Run id: {sink.run_id} ===")

        # 1. global pool (critical, 직렬)
        logger.info("=== [1/5] global_ranking.build_pool ===")
        ok, payload = _run_stage(sink, "global_ranking", build_global_pool, conn, config)
        overall["global_ranking"] = payload
        if not ok:
            logger.error("global_ranking 실패 — 후속 stage 진행 불가. abort.")
            return

        # 2~4. 병렬 (서로 독립, 모두 다른 테이블 write — race 없음)
        logger.info("=== [2-4/5] engagement_aggregator + bandit (member/device) + user_vector — 병렬 ===")
        parallel = [
            ("engagement_aggregator", aggregate_engagement, (conn,), {}),
            ("bandit_reconcile", bandit_reconcile, (conn, config), {}),
            ("bandit_reconcile_devices", bandit_reconcile_devices, (conn, config), {}),
            ("user_vector", build_user_profiles, (conn, config), {}),
        ]
        with ThreadPoolExecutor(max_workers=len(parallel)) as ex:
            futures = {
                ex.submit(_run_stage, sink, name, fn, *args, **kw): name
                for name, fn, args, kw in parallel
            }
            for fut in futures:
                name = futures[fut]
                _, payload = fut.result()
                overall[name] = payload

        # 5. metric rollup (last — reads other stages' outputs)
        logger.info("=== [5/5] metric_rollup ===")
        _, payload = _run_stage(sink, "metric_rollup", metric_rollup, conn, config)
        overall["metric_rollup"] = payload

    finally:
        total_ms = int((time.time() - pipeline_start) * 1000)
        if sink is not None:
            try:
                sink.record("pipeline_total", total_ms, {"stages": list(overall.keys())})
            except Exception:
                logger.debug("pipeline_total record skipped", exc_info=True)
        logger.info(f"=== pipeline_total_ms={total_ms} ===")
        logger.info(f"=== summary: {overall} ===")

        if conn is not None:
            try:
                conn.close()
            except Exception:
                logger.debug("conn close skipped", exc_info=True)


if __name__ == "__main__":
    run_pipeline()
