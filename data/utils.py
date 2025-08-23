import time
import uuid
import numpy as np
from typing import Any, Dict, Sequence
from decimal import Decimal

# 이벤트 가중치(예시) 및 감쇠/윈도우
EVENT_WEIGHTS = {
    "article_in": 1.0,   # 클릭
    "like": 2.0,
    "archive": 2.0,
    "share": 2.0,
    "f_imp": 0.05,
}

TOPK = {
    "cold": {   # 최근 7일 클릭 없음
        "user_embedding": 60,    # A
        "fresh_popular": 80,     # B
        "global_popular": 40,    # C
        "exploration": 20        # D
    },
    "warm": {   # 소수 클릭
        "user_embedding": 100,   # A
        "fresh_popular": 60,     # B
        "global_popular": 20,    # C
        "exploration": 20        # D
    },
    "hot": {    # 활발
        "user_embedding": 120,   # A
        "fresh_popular": 40,     # B
        "global_popular": 25,    # C
        "exploration": 15        # D
    }
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

def _to_py(v):
    """DynamoDB Decimal/중첩 정규화"""
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, dict):
        return {k: _to_py(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_py(x) for x in v]
    return v

def _ensure_keys(d: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    for k in keys:
        if k not in d:
            d[k] = None
    return d

def _query_all_active_users() -> str:
    return f"""
    SELECT member_id
    FROM member
    WHERE
        status='ACTIVE'
        AND role IN ('ROLE_USER', 'ROLE_GUEST')
    """

