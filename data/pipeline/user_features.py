"""
Derive user preference features from engagement data.
Stored as payload in user-profiles Qdrant collection.
"""
import polars as pl
from typing import Dict, Any
from data.utils import USER_COLLECTION
from utils.logger import get_logger

logger = get_logger("UserFeatures")


def compute_user_features(conn) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-user derived features from user_article_engagement.
    Returns {member_id: {feature_name: value}}
    """
    sql = """
    SELECT
        ue.member_id,
        AVG(a.content_length) AS avg_content_length,
        AVG(ue.total_dwell_ms) AS avg_dwell_ms,
        AVG(ue.max_scroll_depth) AS avg_scroll_depth,
        COUNT(DISTINCT a.category_id) AS category_diversity
    FROM user_article_engagement ue
    JOIN article a ON a.article_id = ue.article_id
    WHERE ue.engagement_score > 0
      AND ue.clicks > 0
    GROUP BY ue.member_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return {}

    features: Dict[int, Dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        mid = row["member_id"]
        features[mid] = {
            "pref_content_length": float(row["avg_content_length"] or 0),
            "pref_dwell_ms": float(row["avg_dwell_ms"] or 0),
            "pref_scroll_depth": float(row["avg_scroll_depth"] or 0),
            "category_diversity": int(row["category_diversity"] or 0),
        }
    return features


def compute_user_category_distribution(conn) -> Dict[int, Dict[int, float]]:
    """
    Compute per-user category preference distribution.
    Returns {member_id: {category_id: weight}}
    """
    sql = """
    SELECT
        ue.member_id,
        a.category_id,
        SUM(ue.engagement_score) AS total_score
    FROM user_article_engagement ue
    JOIN article a ON a.article_id = ue.article_id
    WHERE ue.engagement_score > 0
      AND a.category_id IS NOT NULL
    GROUP BY ue.member_id, a.category_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return {}

    result: Dict[int, Dict[int, float]] = {}
    for row in df.iter_rows(named=True):
        mid = row["member_id"]
        cat = row["category_id"]
        score = row["total_score"] or 0.0
        if mid not in result:
            result[mid] = {}
        result[mid][cat] = float(score)

    # Normalize to distribution
    for mid, cats in result.items():
        total = sum(cats.values())
        if total > 0:
            result[mid] = {k: v / total for k, v in cats.items()}

    return result


def update_user_profile_payloads(qdrant, features: Dict[int, Dict[str, Any]]):
    """Update user-profiles payloads with derived features."""
    if not features:
        return

    updated = 0
    for mid, feats in features.items():
        try:
            qdrant.set_payload(collection_name=USER_COLLECTION, payload=feats, points=[mid])
            updated += 1
        except Exception:
            pass

    logger.info(f"Updated {updated}/{len(features)} user profile payloads")
