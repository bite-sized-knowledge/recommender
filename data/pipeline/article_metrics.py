"""
Compute per-article performance metrics from user_article_engagement.
Used by the ranker for popularity-based signals.
"""
import polars as pl
from typing import Dict
from utils.logger import get_logger

logger = get_logger("ArticleMetrics")


def compute_article_metrics(conn) -> pl.DataFrame:
    """
    Compute per-article aggregate metrics.
    Returns DataFrame {article_id, ctr, avg_dwell_ms, avg_scroll_depth, engagement_rate}
    """
    sql = """
    SELECT
        CAST(article_id AS CHAR) AS article_id,
        SUM(impressions) AS total_impressions,
        SUM(clicks) AS total_clicks,
        AVG(CASE WHEN total_dwell_ms > 0 THEN total_dwell_ms END) AS avg_dwell_ms,
        AVG(CASE WHEN max_scroll_depth > 0 THEN max_scroll_depth END) AS avg_scroll_depth,
        SUM(CASE WHEN liked THEN 1 ELSE 0 END) +
        SUM(CASE WHEN bookmarked THEN 1 ELSE 0 END) +
        SUM(CASE WHEN shared THEN 1 ELSE 0 END) AS total_engagements
    FROM user_article_engagement
    WHERE engagement_score IS NOT NULL
    GROUP BY article_id
    HAVING total_impressions >= 5
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={
            "article_id": pl.Utf8,
            "ctr": pl.Float64,
            "avg_dwell_ms": pl.Float64,
            "avg_scroll_depth": pl.Float64,
            "engagement_rate": pl.Float64,
        })

    df = df.with_columns([
        (pl.col("total_clicks").cast(pl.Float64) /
         pl.col("total_impressions").cast(pl.Float64).clip(lower_bound=1)).alias("ctr"),
        (pl.col("total_engagements").cast(pl.Float64) /
         pl.col("total_impressions").cast(pl.Float64).clip(lower_bound=1)).alias("engagement_rate"),
    ])

    return df.select([
        "article_id", "ctr",
        pl.col("avg_dwell_ms").fill_null(0.0),
        pl.col("avg_scroll_depth").fill_null(0.0),
        "engagement_rate",
    ])
