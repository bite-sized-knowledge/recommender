import math
import time
import numpy as np
import polars as pl
from typing import Dict, Set
from collections import defaultdict
from qdrant_client import QdrantClient
from data.utils import _to_point_id, _exp_decay, ITEM_COLLECTION, UUID_NAMESPACE, QDRANT_BATCH
from utils.logger import get_logger

logger = get_logger("Ranker")

W_SIM = 0.70
W_FRESH = 0.15
W_POP = 0.10
W_DIV = 0.05
FRESHNESS_HALF_LIFE_DAYS = 14.0
UNKNOWN_FRESHNESS_DEFAULT = 0.3
SQL_CHUNK = 1000
SECS_PER_DAY = 86400.0


def _fetch_article_metadata(qdrant: QdrantClient, article_ids: list[str]) -> Dict[str, dict]:
    if not article_ids:
        return {}

    id_map = {str(_to_point_id(aid, UUID_NAMESPACE)): aid for aid in article_ids}
    out = {}

    ids = list(id_map.keys())
    for i in range(0, len(ids), QDRANT_BATCH):
        points = qdrant.retrieve(
            collection_name=ITEM_COLLECTION,
            ids=ids[i:i + QDRANT_BATCH],
            with_vectors=False,
            with_payload=True,
        )
        for p in points:
            aid = id_map.get(str(p.id))
            if aid and p.payload:
                out[aid] = p.payload

    return out


def _chunked_query(conn, sql_template: str, ids: list, id_col: str = "id") -> pl.DataFrame:
    """Execute a query with IN clause in chunks to avoid unbounded SQL."""
    frames = []
    for i in range(0, len(ids), SQL_CHUNK):
        chunk = ids[i:i + SQL_CHUNK]
        placeholders = ",".join(str(int(x)) if isinstance(x, int) else f"'{x}'" for x in chunk)
        sql = sql_template.format(placeholders=placeholders)
        df = conn.execute(sql)
        if not df.is_empty():
            frames.append(df)
    return pl.concat(frames) if frames else pl.DataFrame()


def _fetch_read_articles(conn, member_ids: list[int]) -> Dict[int, Set[str]]:
    if not member_ids:
        return {}

    sql_tpl = """
    SELECT member_id, CAST(article_id AS CHAR) AS article_id
    FROM article_history
    WHERE member_id IN ({placeholders})
    """
    df = _chunked_query(conn, sql_tpl, member_ids)
    if df.is_empty():
        return {}

    out: Dict[int, Set[str]] = defaultdict(set)
    for mid, aid in df.select(["member_id", "article_id"]).iter_rows():
        out[mid].add(aid)
    return dict(out)


def _fetch_article_popularity(conn, article_ids: list[str]) -> Dict[str, float]:
    if not article_ids:
        return {}

    sql_tpl = """
    SELECT CAST(article_id AS CHAR) AS article_id,
           AVG(engagement_score) AS avg_score
    FROM user_article_engagement
    WHERE article_id IN ({placeholders})
      AND engagement_score IS NOT NULL
    GROUP BY article_id
    """
    df = _chunked_query(conn, sql_tpl, article_ids)
    if df.is_empty():
        return {}

    return {row["article_id"]: row["avg_score"] for row in df.iter_rows(named=True)}


def rank_candidates(candidates: pl.DataFrame, conn, qdrant: QdrantClient) -> pl.DataFrame:
    """
    Re-rank candidates using multi-signal scoring.
    Input/Output: DataFrame {member_id, article_id, score, source}
    """
    if candidates.is_empty():
        return candidates

    now_epoch = time.time()
    unique_articles = candidates["article_id"].unique().to_list()
    unique_members = candidates["member_id"].unique().to_list()

    logger.info(f"Fetching metadata for {len(unique_articles)} articles")
    article_meta = _fetch_article_metadata(qdrant, unique_articles)
    read_history = _fetch_read_articles(conn, unique_members)
    popularity = _fetch_article_popularity(conn, unique_articles)

    user_category_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    scored_rows = []
    for row in candidates.iter_rows(named=True):
        mid = row["member_id"]
        aid = row["article_id"]
        sim_score = row.get("score", 0.0) or 0.0

        if mid in read_history and aid in read_history[mid]:
            scored_rows.append({**row, "score": -1.0})
            continue

        meta = article_meta.get(aid, {})

        pub_epoch = meta.get("published_at", 0) or 0
        if pub_epoch <= 0:
            freshness = UNKNOWN_FRESHNESS_DEFAULT
        else:
            days_ago = max(0.0, (now_epoch - pub_epoch) / SECS_PER_DAY)
            freshness = _exp_decay(days_ago, FRESHNESS_HALF_LIFE_DAYS)

        quality = meta.get("quality_score", 3) or 3
        quality_bonus = 0.05 if quality >= 4 else (-0.05 if quality <= 2 else 0.0)

        pop_score = popularity.get(aid, 0.0) or 0.0
        pop_norm = math.log1p(max(0, pop_score)) / 5.0

        cat = meta.get("category", 0) or 0
        user_category_counts[mid][cat] += 1
        div_bonus = 1.0 / (1.0 + user_category_counts[mid][cat] * 0.1)

        final_score = (
            W_SIM * sim_score
            + W_FRESH * freshness
            + W_POP * pop_norm
            + W_DIV * div_bonus
            + quality_bonus
        )

        scored_rows.append({**row, "score": final_score})

    result = pl.from_dicts(scored_rows, schema=candidates.schema)
    result = (
        result
        .filter(pl.col("score") > 0)
        .sort(["member_id", "score"], descending=[False, True])
    )

    logger.info(f"Ranking complete: {len(result)} candidates after filtering")
    return result
