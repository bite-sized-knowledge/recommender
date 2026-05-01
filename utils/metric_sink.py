"""recommender 배치 stage 메트릭 적재 (recommender_run_metric).

각 stage 의 시간/payload 를 기록한다. 적재 실패는 graceful skip — 본 파이프라인을 막지 않는다.
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
