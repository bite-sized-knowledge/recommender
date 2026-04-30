"""
Phase 1: 글로벌 article 풀 빌드.

freshness × quality × popularity 의 가중합으로 매일 새 풀을 만들어
recommendation_global 테이블에 통째로 swap한다 (TRUNCATE + INSERT in tx).

유저 0인 환경에서도 동작:
- popularity 0이면 freshness + quality 만으로 score 계산.
- category_id NULL 글은 풀에서 제외 (bandit이 카테고리 단위라 매칭 불가).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import polars as pl
from sqlalchemy import text

from utils.logger import get_logger

logger = get_logger("GlobalRanking")


def _fetch_articles(conn, recency_days: int) -> pl.DataFrame:
    sql = f"""
    SELECT
        CAST(a.article_id AS CHAR)              AS article_id,
        a.category_id                            AS category_id,
        a.published_at                           AS published_at,
        COALESCE(a.quality_score, 5)             AS quality_score,
        COALESCE(a.like_count, 0)                AS like_count,
        COALESCE(a.share_count, 0)               AS share_count,
        COALESCE(a.bookmark_count, 0)            AS bookmark_count,
        TIMESTAMPDIFF(HOUR, a.published_at, NOW()) / 24.0 AS days_old
    FROM article a
    WHERE a.published_at IS NOT NULL
      AND a.category_id IS NOT NULL
      AND a.published_at >= NOW() - INTERVAL {int(recency_days)} DAY
    """
    return conn.execute(sql)


def _fetch_recent_clicks(conn, window_days: int) -> pl.DataFrame:
    """user_events에서 article_in 클릭 수 (popularity 신호 보강)."""
    sql = f"""
    SELECT
        CAST(article_id AS CHAR) AS article_id,
        COUNT(*)                  AS recent_clicks
    FROM user_events
    WHERE LOWER(event_type) = 'article_in'
      AND article_id IS NOT NULL
      AND occurred_at >= NOW() - INTERVAL {int(window_days)} DAY
    GROUP BY article_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={"article_id": pl.Utf8, "recent_clicks": pl.Int64})
    return df


def _compute_scores(df: pl.DataFrame, half_life_days: float, weights: Dict[str, float]) -> pl.DataFrame:
    w_fresh = float(weights.get("freshness", 0.4))
    w_qual = float(weights.get("quality", 0.4))
    w_pop = float(weights.get("popularity", 0.2))

    pop_log = (
        pl.col("like_count").cast(pl.Float64).log1p()
        + 2.0 * pl.col("bookmark_count").cast(pl.Float64).log1p()
        + 2.0 * pl.col("share_count").cast(pl.Float64).log1p()
        + pl.col("recent_clicks").cast(pl.Float64).log1p()
    )

    df = df.with_columns([
        pl.col("recent_clicks").fill_null(0),
        (0.5 ** (pl.col("days_old").cast(pl.Float64) / float(half_life_days))).alias("freshness"),
        (pl.col("quality_score").cast(pl.Float64) / 10.0).alias("quality_norm"),
        pop_log.alias("pop_raw"),
    ])

    pop_max = float(df["pop_raw"].max() or 1.0)
    if pop_max <= 0:
        pop_max = 1.0

    df = df.with_columns([
        (pl.col("pop_raw") / pop_max).alias("popularity_norm"),
    ])

    df = df.with_columns([
        (
            w_fresh * pl.col("freshness")
            + w_qual * pl.col("quality_norm")
            + w_pop * pl.col("popularity_norm")
        ).alias("score")
    ])

    return df.select(["article_id", "category_id", "score", "freshness", "quality_norm", "popularity_norm", "days_old"])


def _enforce_per_category_min(df: pl.DataFrame, pool_size: int, per_category_min: int) -> pl.DataFrame:
    """
    Long-tail 카테고리 보장: 각 카테고리에서 최소 N개를 score 순으로 먼저 뽑고,
    남은 슬롯은 글로벌 score 순으로 채운다.
    """
    if df.is_empty():
        return df

    df = df.sort("score", descending=True)

    if per_category_min <= 0:
        return df.head(pool_size)

    # category별 top-N
    head = df.group_by("category_id", maintain_order=True).head(per_category_min)
    selected_ids = set(head["article_id"].to_list())

    # 남은 슬롯
    remaining_slots = pool_size - len(head)
    if remaining_slots <= 0:
        return head.head(pool_size)

    rest = df.filter(~pl.col("article_id").is_in(list(selected_ids))).head(remaining_slots)
    combined = pl.concat([head, rest], how="vertical")
    return combined.sort("score", descending=True).head(pool_size)


def _swap_pool(conn, df: pl.DataFrame) -> int:
    """recommendation_global 통째 swap: TRUNCATE + batch INSERT in single transaction."""
    if df.is_empty():
        logger.warning("글로벌 풀이 비어있음. swap 생략")
        return 0

    rows = [
        {
            "article_id": r["article_id"],
            "score": float(r["score"]),
            "rank_global": int(r["rank_global"]),
            "category_id": int(r["category_id"]) if r["category_id"] is not None else None,
        }
        for r in df.iter_rows(named=True)
    ]

    insert_sql = """
        INSERT INTO recommendation_global
            (article_id, score, rank_global, category_id, generated_at)
        VALUES (:article_id, :score, :rank_global, :category_id, NOW())
    """

    with conn.engine.connect() as c:
        with c.begin():
            c.execute(text("TRUNCATE TABLE recommendation_global"))
            batch_size = 500
            for i in range(0, len(rows), batch_size):
                c.execute(text(insert_sql), rows[i : i + batch_size])

    return len(rows)


def build_pool(conn, config: Dict) -> Dict:
    """
    글로벌 풀 빌드 + swap. main pipeline의 첫 stage에서 호출.
    Returns metric payload (pool_size, per-category dist, score stats).
    """
    cfg = config.get("global_ranking", {})
    recency_days = int(cfg.get("recency_days", 60))
    half_life = float(cfg.get("freshness_half_life_days", 14))
    pool_size = int(cfg.get("pool_size", 1000))
    per_category_min = int(cfg.get("per_category_min", 30))
    pop_window = int(cfg.get("popularity_window_days", 7))
    weights = cfg.get("weights", {})

    articles = _fetch_articles(conn, recency_days)
    clicks = _fetch_recent_clicks(conn, pop_window)

    if articles.is_empty():
        logger.warning("article 후보 0건 — 풀 빌드 생략")
        return {"pool_size": 0, "candidates": 0}

    df = articles.join(clicks, on="article_id", how="left")
    df = _compute_scores(df, half_life, weights)
    df = _enforce_per_category_min(df, pool_size, per_category_min)

    df = df.sort("score", descending=True).with_row_index("rank_global", offset=1)

    inserted = _swap_pool(conn, df)

    payload: Dict = {
        "pool_size": inserted,
        "candidates": int(len(articles)),
    }
    if not df.is_empty():
        payload["score_min"] = round(float(df["score"].min()), 4)
        payload["score_max"] = round(float(df["score"].max()), 4)
        payload["score_mean"] = round(float(df["score"].mean()), 4)
        per_cat = (
            df.group_by("category_id")
            .len()
            .sort("len", descending=True)
            .head(20)
        )
        payload["per_category_top20"] = {
            int(r["category_id"]): int(r["len"])
            for r in per_cat.iter_rows(named=True)
        }
        payload["categories_total"] = int(df["category_id"].n_unique())
    logger.info(f"global_ranking.build_pool done: {payload}")
    return payload
