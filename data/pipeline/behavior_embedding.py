import uuid
import numpy as np
import polars as pl
from typing import Dict, Optional, List
from data.utils import (
    _now_ms, _exp_decay, _to_point_id, _l2_normalize,
    _query_all_active_users, EVENT_WEIGHTS
)

# Config
ITEM_COLLECTION = "bite-vectordb"
USER_COLLECTION = "user-profiles"
UUID_NAMESPACE = uuid.NAMESPACE_DNS


HALF_LIFE_DAYS = 3.0
LOOKBACK_DAYS = 7
TOPN_ARTICLES = 50

def _events_to_pl(events: list[dict]) -> pl.DataFrame:
    """
    compute_alpha에서 요구하는 최소 컬럼만 정규화
    - event_type: str
    - timestamp: int(ms)
    """
    if not events:
        return pl.DataFrame(schema={"event_type": pl.Utf8, "timestamp": pl.Int64})
    df = pl.DataFrame({
        "event_type": [str(e.get("event_type", "")) for e in events],
        "timestamp":  [int(e.get("timestamp", 0)) for e in events],
    })
    # 결측/이상치 방어
    return df.filter(pl.col("event_type") != "").filter(pl.col("timestamp") > 0)


def fetch_recent_events(conn, member_id: int, lookback_days: int = LOOKBACK_DAYS) -> List[Dict]:
    sql = f"""
    SELECT
        LOWER(event_type) AS event_type,
        CAST(article_id AS CHAR) AS target_id,
        UNIX_TIMESTAMP(occurred_at) * 1000 AS timestamp
    FROM user_events
    WHERE member_id = {member_id}
      AND occurred_at >= NOW() - INTERVAL {lookback_days} DAY
      AND article_id IS NOT NULL
    """

    df = conn.execute(sql)
    if df.is_empty():
        return []

    return df.to_dicts()

def aggregate_article_weights(events: List[Dict]) -> Dict[str, float]:
    if not events:
        return {}

    now_ms = _now_ms()
    ms_per_day = 24 * 60 * 60 * 1000
    weights: Dict[str, float] = {}

    for e in events:
        et = e.get("event_type")
        w_type = EVENT_WEIGHTS.get(et, 0.0)
        if w_type == 0:
            continue

        ts = e.get("timestamp")
        if ts is None:
            continue

        days_ago = max(0.0, (now_ms - int(ts)) / ms_per_day)
        w = w_type * _exp_decay(days_ago, HALF_LIFE_DAYS)

        aid = str(e["target_id"])
        weights[aid] = weights.get(aid, 0.0) + w

    # 상위 N개만 사용
    if len(weights) > TOPN_ARTICLES:
        weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:TOPN_ARTICLES])
    return weights

def fetch_vectors_by_article_ids(
        qdrant,
        article_ids: List[str],
        batch: int = 1024
    ) -> Dict[str, np.ndarray]:
    """
    article_id 리스트를 받아 uuid5로 포인트 ID를 생성하고 retrieve로 벡터를 조회.
    반환: {article_id: vector}
    """
    if not article_ids:
        return {}

    id_map: Dict[str, str] = {str(_to_point_id(aid, UUID_NAMESPACE)): aid for aid in article_ids}

    out: Dict[str, np.ndarray] = {}

    ids = list(id_map.keys())
    for i in range(0, len(ids), batch):
        chunk_ids = ids[i:i+batch]

        points = qdrant.retrieve(
            collection_name=ITEM_COLLECTION,
            ids=chunk_ids,
            with_vectors=True,
            with_payload=False,
        )

        for p in points:
            pid = str(p.id)
            aid = id_map.get(pid)
            if aid is None:
                continue
            if p.vector is None:
                continue
            out[aid] = np.asarray(p.vector, dtype=np.float32)

    return out

def build_user_behavior_embedding(
        qdrant,
        conn,
        member_id: int,
        lookback_days: int = LOOKBACK_DAYS
    ) -> Optional[np.ndarray]:
    """
    단일 유저에 대한 behavior embedding 생성
    """

    # (1) 이벤트 조회
    events = fetch_recent_events(conn, member_id, lookback_days)
    user_logs = _events_to_pl(events)
    if not events:
        return None, user_logs

    # (2) 가중치 집계
    w_by_article = aggregate_article_weights(events)
    if not w_by_article:
        return None, user_logs

    # (3) Qdrant에서 벡터 조회 (ID 기반 retrieve)
    vec_map = fetch_vectors_by_article_ids(qdrant, list(w_by_article.keys()))
    if not vec_map:
        return None, user_logs

    # (4) 가중 평균
    vecs, ws = [], []
    for aid, w in w_by_article.items():
        v = vec_map.get(aid)
        if v is not None and np.isfinite(w) and w > 0:
            vecs.append(v)
            ws.append(float(w))

    if not vecs:
        return None, user_logs

    V = np.vstack(vecs).astype(np.float32)       # [N, D]
    W = np.asarray(ws, dtype=np.float32)         # [N]
    if not np.isfinite(W).all() or W.sum() <= 0:
        return None, user_logs

    centroid = (V * W[:, None]).sum(axis=0) / (W.sum() + 1e-9)
    return _l2_normalize(centroid), user_logs


def build_behavior_embedding(
        conn,
        qdrant,
    ) -> Dict[int, dict]:
    """
    모든 유저에 대해 behavior embedding 계산
    """

    res = {}

    active_users = conn.execute(_query_all_active_users())
    for user in active_users['member_id']:
        vector, user_logs = build_user_behavior_embedding(
            qdrant=qdrant,
            conn=conn,
            member_id=user
        )

        res[user] = {"vector" : vector, "logs" : user_logs}

    return res
