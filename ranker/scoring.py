import math
import time
import uuid
import numpy as np
import polars as pl
from typing import Dict, Set
from qdrant_client import QdrantClient
from data.utils import _to_point_id
from utils.logger import get_logger

logger = get_logger("Ranker")

ITEM_COLLECTION = "bite-vectordb"
UUID_NAMESPACE = uuid.NAMESPACE_DNS

# Scoring weights
W_SIM = 0.70
W_FRESH = 0.15
W_POP = 0.10
W_DIV = 0.05
FRESHNESS_HALF_LIFE_DAYS = 14.0


def _freshness_decay(published_epoch: float, now_epoch: float, half_life_days: float = FRESHNESS_HALF_LIFE_DAYS) -> float:
    if published_epoch <= 0:
        return 0.3  # unknown publish date gets a low default
    days_ago = max(0.0, (now_epoch - published_epoch) / 86400.0)
    return 0.5 ** (days_ago / half_life_days)


def _fetch_article_metadata(qdrant: QdrantClient, article_ids: list[str]) -> Dict[str, dict]:
    """Fetch published_at, quality_score, category from Qdrant payload."""
    if not article_ids:
        return {}

    id_map = {str(_to_point_id(aid, UUID_NAMESPACE)): aid for aid in article_ids}
    out = {}

    ids = list(id_map.keys())
    for i in range(0, len(ids), 256):
        chunk = ids[i:i+256]
        points = qdrant.retrieve(
            collection_name=ITEM_COLLECTION,
            ids=chunk,
            with_vectors=False,
            with_payload=True,
        )
        for p in points:
            aid = id_map.get(str(p.id))
            if aid and p.payload:
                out[aid] = p.payload

    return out


def _fetch_read_articles(conn, member_ids: list[int]) -> Dict[int, Set[str]]:
    """Fetch article_history for given users."""
    if not member_ids:
        return {}

    placeholders = ",".join(str(int(m)) for m in member_ids)
    sql = f"""
    SELECT member_id, CAST(article_id AS CHAR) AS article_id
    FROM article_history
    WHERE member_id IN ({placeholders})
    """
    df = conn.execute(sql)
    if df.is_empty():
        return {}

    out: Dict[int, Set[str]] = {}
    for row in df.iter_rows(named=True):
        mid = row["member_id"]
        if mid not in out:
            out[mid] = set()
        out[mid].add(row["article_id"])
    return out


def _fetch_article_popularity(conn, article_ids: list[str]) -> Dict[str, float]:
    """Fetch engagement_score from user_article_engagement, aggregated per article."""
    if not article_ids:
        return {}

    placeholders = ",".join(f"'{aid}'" for aid in article_ids)
    sql = f"""
    SELECT CAST(article_id AS CHAR) AS article_id,
           AVG(engagement_score) AS avg_score
    FROM user_article_engagement
    WHERE article_id IN ({placeholders})
      AND engagement_score IS NOT NULL
    GROUP BY article_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return {}

    return {row["article_id"]: row["avg_score"] for row in df.iter_rows(named=True)}


def rank_candidates(candidates: pl.DataFrame, conn, qdrant: QdrantClient) -> pl.DataFrame:
    """
    Re-rank candidates using multi-signal scoring.

    Input DataFrame: {member_id, article_id, score, source}
    Output DataFrame: {member_id, article_id, score, source} with updated scores, sorted per user.
    """
    if candidates.is_empty():
        return candidates

    now_epoch = time.time()

    # Collect unique article_ids and member_ids
    unique_articles = candidates["article_id"].unique().to_list()
    unique_members = candidates["member_id"].unique().to_list()

    # Batch fetch metadata
    logger.info(f"Fetching metadata for {len(unique_articles)} articles")
    article_meta = _fetch_article_metadata(qdrant, unique_articles)
    read_history = _fetch_read_articles(conn, unique_members)
    popularity = _fetch_article_popularity(conn, unique_articles)

    # Track category diversity per user
    user_category_counts: Dict[int, Dict[int, int]] = {}

    # Score each row
    scored_rows = []
    for row in candidates.iter_rows(named=True):
        mid = row["member_id"]
        aid = row["article_id"]
        sim_score = row.get("score", 0.0) or 0.0

        # Already-seen penalty
        if mid in read_history and aid in read_history[mid]:
            scored_rows.append({**row, "score": -1.0})
            continue

        meta = article_meta.get(aid, {})

        # Freshness
        pub_epoch = meta.get("published_at", 0) or 0
        freshness = _freshness_decay(pub_epoch, now_epoch)

        # Quality bonus
        quality = meta.get("quality_score", 3) or 3
        quality_bonus = 0.05 if quality >= 4 else (-0.05 if quality <= 2 else 0.0)

        # Popularity
        pop_score = popularity.get(aid, 0.0) or 0.0
        pop_norm = math.log1p(max(0, pop_score)) / 5.0  # normalize to ~0-1 range

        # Category diversity bonus
        cat = meta.get("category", 0) or 0
        if mid not in user_category_counts:
            user_category_counts[mid] = {}
        cat_counts = user_category_counts[mid]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        div_bonus = 1.0 / (1.0 + cat_counts.get(cat, 0) * 0.1)

        final_score = (
            W_SIM * sim_score
            + W_FRESH * freshness
            + W_POP * pop_norm
            + W_DIV * div_bonus
            + quality_bonus
        )

        scored_rows.append({**row, "score": final_score})

    result = pl.from_dicts(scored_rows, schema=candidates.schema)

    # Sort per user by score DESC, filter out already-read
    result = (
        result
        .filter(pl.col("score") > 0)
        .sort(["member_id", "score"], descending=[False, True])
    )

    logger.info(f"Ranking complete: {len(result)} candidates after filtering")
    return result
