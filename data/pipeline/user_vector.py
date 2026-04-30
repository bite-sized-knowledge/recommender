"""
Phase 2: 유저별 임베딩 EMA 빌드 → Qdrant user_profile collection upsert.

각 유저의 클릭한 article 임베딩들을 시간 가중 평균하여 user vector 산출.
weight = ema_decay ^ days_ago (예: 0.95^days)

유저 0 또는 클릭 0인 환경에서는 빈 결과 — 정상.
서빙 측은 user_profile 없으면 글로벌 score fallback.

Article 임베딩은 Qdrant `bite-vectordb` collection 에 이미 저장됨.
point_id 매핑은 article_id 기반 uuid5 (data.utils._to_point_id 와 동일 규칙).
"""
from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from data.utils import UUID_NAMESPACE, ITEM_COLLECTION, _l2_normalize, _to_point_id
from utils.logger import get_logger

logger = get_logger("UserVector")


def _ensure_collection(client, collection: str, vector_dim: int) -> None:
    """user_profile collection이 없으면 생성 (cosine similarity)."""
    try:
        existing = client.get_collections().collections
        names = {c.name for c in existing}
    except Exception:
        names = set()

    if collection in names:
        return

    from qdrant_client.http import models as qmodels
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=int(vector_dim),
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info(f"qdrant collection 생성: {collection} (dim={vector_dim})")


def _fetch_user_clicks(conn, lookback_days: int) -> pl.DataFrame:
    """
    user_events 에서 article_in / like / archive / share 이벤트 → (member_id, article_id, days_ago)
    동일 (member, article) 다수 이벤트면 가장 최근 occurred_at 사용.
    """
    sql = f"""
    SELECT
        e.member_id,
        CAST(e.article_id AS CHAR) AS article_id,
        TIMESTAMPDIFF(HOUR, MAX(e.occurred_at), NOW()) / 24.0 AS days_ago
    FROM user_events e
    WHERE e.member_id IS NOT NULL
      AND e.article_id IS NOT NULL
      AND e.occurred_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND LOWER(e.event_type) IN ('article_in','like','archive','share')
    GROUP BY e.member_id, e.article_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={
            "member_id": pl.Int64, "article_id": pl.Utf8, "days_ago": pl.Float64,
        })
    return df


def _retrieve_article_vectors(client, article_ids: List[str]) -> Dict[str, np.ndarray]:
    """Qdrant `bite-vectordb` 에서 article 임베딩 retrieve."""
    if not article_ids:
        return {}

    point_ids = [str(_to_point_id(aid, UUID_NAMESPACE)) for aid in article_ids]
    id_to_aid = dict(zip(point_ids, article_ids))

    out: Dict[str, np.ndarray] = {}
    batch = 256
    for i in range(0, len(point_ids), batch):
        chunk = point_ids[i : i + batch]
        try:
            points = client.retrieve(
                collection_name=ITEM_COLLECTION,
                ids=chunk,
                with_vectors=True,
                with_payload=False,
            )
        except Exception as e:
            logger.warning(f"qdrant retrieve 실패 batch[{i}:{i+batch}]: {e}")
            continue
        for p in points:
            if p.vector is None:
                continue
            aid = id_to_aid.get(str(p.id))
            if aid is None:
                continue
            out[aid] = np.asarray(p.vector, dtype=np.float32)
    return out


def _build_user_vector(
    rows: pl.DataFrame,
    article_vecs: Dict[str, np.ndarray],
    decay: float,
) -> Optional[np.ndarray]:
    """시간 가중 평균. weight = decay ^ days_ago."""
    weights = []
    vecs = []
    for r in rows.iter_rows(named=True):
        v = article_vecs.get(r["article_id"])
        if v is None:
            continue
        w = float(decay) ** max(0.0, float(r["days_ago"]))
        weights.append(w)
        vecs.append(v)

    if not vecs:
        return None

    W = np.asarray(weights, dtype=np.float32)[:, None]
    V = np.vstack(vecs).astype(np.float32)
    weighted_sum = (V * W).sum(axis=0)
    total_w = float(W.sum())
    if total_w <= 0:
        return None
    avg = weighted_sum / total_w
    return _l2_normalize(avg.astype(np.float32))


def build_profiles(conn, config: Dict) -> Dict:
    """
    Phase 2 stage. 모든 유저의 클릭 임베딩 EMA → user_profile collection upsert.
    """
    cfg = config.get("user_vector", {})
    if not cfg.get("enabled", True):
        return {"skipped": True}

    collection = str(cfg.get("collection", "user_profile"))
    vector_dim = int(cfg.get("vector_dim", 1024))
    decay = float(cfg.get("ema_decay", 0.95))
    min_clicks = int(cfg.get("min_clicks", 1))
    batch_size = int(cfg.get("qdrant_batch", 256))
    lookback = int(cfg.get("lookback_days", 90))

    clicks = _fetch_user_clicks(conn, lookback)
    if clicks.is_empty():
        logger.info("user_vector: 클릭 이벤트 0건 — 빌드 생략")
        return {"users_built": 0, "users_total": 0, "events_total": 0}

    counts = clicks.group_by("member_id").len().rename({"len": "click_count"})
    eligible = counts.filter(pl.col("click_count") >= min_clicks)
    if eligible.is_empty():
        return {"users_built": 0, "users_total": int(len(counts)), "events_total": int(len(clicks))}

    clicks = clicks.join(eligible.select("member_id"), on="member_id", how="inner")
    article_ids = clicks["article_id"].unique().to_list()

    qdrant = conn.get_qdrant()
    _ensure_collection(qdrant, collection, vector_dim)
    article_vecs = _retrieve_article_vectors(qdrant, article_ids)

    if not article_vecs:
        logger.warning("user_vector: article 임베딩 0건 retrieve — Qdrant bite-vectordb 점검 필요")
        return {"users_built": 0, "users_total": int(len(eligible)), "events_total": int(len(clicks)), "article_vecs_retrieved": 0}

    from qdrant_client.http import models as qmodels

    points: List[qmodels.PointStruct] = []
    built = 0
    for member_id, group in clicks.group_by("member_id"):
        mid = int(member_id[0]) if isinstance(member_id, tuple) else int(member_id)
        v = _build_user_vector(group, article_vecs, decay)
        if v is None:
            continue
        points.append(qmodels.PointStruct(
            id=mid,
            vector=v.tolist(),
            payload={
                "member_id": mid,
                "click_count": int(len(group)),
            },
        ))
        built += 1

    if points:
        for i in range(0, len(points), batch_size):
            try:
                qdrant.upsert(
                    collection_name=collection,
                    points=points[i : i + batch_size],
                )
            except Exception as e:
                logger.warning(f"qdrant upsert 실패 batch[{i}:{i+batch_size}]: {e}")

    payload = {
        "users_built": built,
        "users_total": int(len(eligible)),
        "events_total": int(len(clicks)),
        "article_vecs_retrieved": int(len(article_vecs)),
        "article_vecs_requested": int(len(article_ids)),
    }
    logger.info(f"user_vector.build_profiles done: {payload}")
    return payload
