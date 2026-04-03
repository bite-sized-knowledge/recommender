import time
import uuid
import numpy as np
from typing import Dict

# 이벤트 가중치(예시) 및 감쇠/윈도우
EVENT_WEIGHTS = {
    "article_in": 1.0,   # 클릭
    "like": 2.0,
    "archive": 2.0,
    "share": 2.0,
    "f_imp": 0.05,
    "uninterest": -3.0,  # 부정 피드백
}

def _now_ms() -> int:
    return int(time.time() * 1000)

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (v / n).astype(np.float32)

def _exp_decay(days_ago: float, half_life_days: float) -> float:
    return 0.5 ** (days_ago / half_life_days)

def _to_point_id(article_id: str, namespace) -> uuid.UUID:
    return uuid.uuid5(namespace, article_id)

def _query_all_active_users() -> str:
    return f"""
    SELECT member_id
    FROM member
    WHERE
        status='ACTIVE'
        AND role IN ('ROLE_USER', 'ROLE_GUEST')
    """

