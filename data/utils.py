import time
import uuid
import numpy as np
from typing import Dict

# Shared constants
ITEM_COLLECTION = "bite-vectordb"
USER_COLLECTION = "user-profiles"
CAT_COLLECTION = "category-profiles"
UUID_NAMESPACE = uuid.NAMESPACE_DNS
QDRANT_BATCH = 256
RETENTION_DAYS = 180

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

def _scroll_centroid(client, collection: str, scroll_filter, min_vecs: int = 1):
    """Scroll a Qdrant collection with a filter and return the L2-normalized centroid vector."""
    vecs = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            with_vectors=True,
            with_payload=False,
            offset=next_offset,
            scroll_filter=scroll_filter,
        )
        for p in points:
            if p.vector is not None:
                vecs.append(np.asarray(p.vector, dtype=np.float32))
        if next_offset is None:
            break

    if len(vecs) < min_vecs:
        return None
    return _l2_normalize(np.vstack(vecs).mean(axis=0).astype(np.float32))

