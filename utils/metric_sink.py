"""metric.bite-sized.xyz 대시보드용 배치 메트릭 적재.

run_pipeline()의 stage별 메트릭을 recommender_run_metric에,
RecallEvaluator의 일별 결과를 recommender_recall_daily에 기록한다.
적재 실패는 모두 graceful skip — 본 파이프라인을 막지 않는다.
"""

from __future__ import annotations

import json
import logging
import uuid

logger = logging.getLogger(__name__)


class MetricSink:
    def __init__(self, conn):
        self.conn = conn
        self.run_id = str(uuid.uuid4())

    def record(self, stage, duration_ms=None, payload=None):
        try:
            self.conn._raw_execute(
                """
                INSERT INTO recommender_run_metric
                    (run_id, stage, duration_ms, payload)
                VALUES (:run_id, :stage, :duration_ms, :payload)
                """,
                {
                    "run_id": self.run_id,
                    "stage": stage,
                    "duration_ms": duration_ms,
                    "payload": (
                        json.dumps(payload, ensure_ascii=False, default=str)
                        if payload
                        else None
                    ),
                },
            )
        except Exception:
            logger.debug("metric_sink record skipped", exc_info=True)


def upsert_recall_daily(
    conn,
    metric_date,
    k,
    recall,
    hit_users=None,
    total_users=None,
    total_recommendations=None,
    unique_items=None,
):
    try:
        conn._raw_execute(
            """
            INSERT INTO recommender_recall_daily
                (metric_date, k, recall, hit_users, total_users,
                 total_recommendations, unique_items)
            VALUES
                (:metric_date, :k, :recall, :hit_users, :total_users,
                 :total_recommendations, :unique_items)
            ON DUPLICATE KEY UPDATE
                recall = VALUES(recall),
                hit_users = VALUES(hit_users),
                total_users = VALUES(total_users),
                total_recommendations = VALUES(total_recommendations),
                unique_items = VALUES(unique_items),
                updated_at = CURRENT_TIMESTAMP
            """,
            {
                "metric_date": metric_date,
                "k": k,
                "recall": recall,
                "hit_users": hit_users,
                "total_users": total_users,
                "total_recommendations": total_recommendations,
                "unique_items": unique_items,
            },
        )
    except Exception:
        logger.debug("recall_daily upsert skipped", exc_info=True)
