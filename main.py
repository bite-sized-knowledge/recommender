"""
recommender 배치 파이프라인 — Phase 1+2.

Stage:
  1. global_ranking.build_pool        : recommendation_global atomic swap
  2. engagement_aggregator            : user_events → user_article_engagement upsert (보조)
  3. bandit_reconcile.reconcile       : member_category_bandit ground-truth overwrite
  4. user_vector.build_profiles       : Phase 2 — Qdrant user_profile EMA upsert
  5. metric_rollup.rollup             : recommendation_metric_daily + bandit_state_snapshot

각 stage 시간/payload 는 recommender_run_metric 에 저장.
실패한 stage 는 graceful skip — 다음 stage 계속 (단, 글로벌 풀은 critical → 실패 시 abort).
"""
from __future__ import annotations

import time

from common.db import Connection
from data.pipeline.bandit_reconcile import reconcile as bandit_reconcile
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

        # 1. global pool (critical)
        logger.info("=== [1/5] global_ranking.build_pool ===")
        ok, payload = _run_stage(sink, "global_ranking", build_global_pool, conn, config)
        overall["global_ranking"] = payload
        if not ok:
            logger.error("global_ranking 실패 — 후속 stage 진행 불가. abort.")
            return

        # 2. engagement aggregator (best-effort)
        logger.info("=== [2/5] engagement_aggregator ===")
        _, payload = _run_stage(sink, "engagement_aggregator", aggregate_engagement, conn)
        overall["engagement_aggregator"] = payload

        # 3. bandit reconcile
        logger.info("=== [3/5] bandit_reconcile ===")
        _, payload = _run_stage(sink, "bandit_reconcile", bandit_reconcile, conn, config)
        overall["bandit_reconcile"] = payload

        # 4. user vector (Phase 2)
        logger.info("=== [4/5] user_vector.build_profiles ===")
        _, payload = _run_stage(sink, "user_vector", build_user_profiles, conn, config)
        overall["user_vector"] = payload

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
                pass
        logger.info(f"=== pipeline_total_ms={total_ms} ===")
        logger.info(f"=== summary: {overall} ===")

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    run_pipeline()
