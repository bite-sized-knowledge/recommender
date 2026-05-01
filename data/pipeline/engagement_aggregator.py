from sqlalchemy import text

from utils.logger import get_logger

logger = get_logger("EngagementAggregator")

# MAX(CASE WHEN ... THEN occurred_at END) 빈 그룹의 cast 가 NO_ZERO_DATE/NO_ZERO_IN_DATE 를
# 포함한 sql_mode 에서 reject. SQLAlchemy connect-event 의 session-level SET 만으로는 일부
# connection 에 안 적용되는 케이스가 관측되어 같은 connection 안에서 statement-level 명시.
_RELAXED_SQL_MODE = "STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION"


def aggregate_engagement(conn) -> None:
    """user_events 의 유저×아티클 engagement 를 집계해서 user_article_engagement 에 UPSERT."""
    sql = """
    INSERT INTO user_article_engagement
        (member_id, article_id, impressions, clicks, total_dwell_ms, max_scroll_depth,
         bookmarked, liked, shared, first_seen_at, last_seen_at, engagement_score)
    SELECT
        member_id,
        article_id,
        SUM(CASE WHEN LOWER(event_type) = 'f_imp' THEN 1 ELSE 0 END) AS impressions,
        SUM(CASE WHEN LOWER(event_type) = 'article_in' THEN 1 ELSE 0 END) AS clicks,
        COALESCE(SUM(dwell_time_ms), 0) AS total_dwell_ms,
        COALESCE(MAX(scroll_depth), 0) AS max_scroll_depth,
        CASE WHEN COALESCE(NULLIF(MAX(CASE WHEN LOWER(event_type) = 'archive' THEN occurred_at END), '0000-00-00 00:00:00'), '1970-01-01')
                  > COALESCE(NULLIF(MAX(CASE WHEN LOWER(event_type) = 'archive_cancel' THEN occurred_at END), '0000-00-00 00:00:00'), '1970-01-01')
             THEN 1 ELSE 0 END AS bookmarked,
        CASE WHEN COALESCE(NULLIF(MAX(CASE WHEN LOWER(event_type) = 'like' THEN occurred_at END), '0000-00-00 00:00:00'), '1970-01-01')
                  > COALESCE(NULLIF(MAX(CASE WHEN LOWER(event_type) = 'like_cancel' THEN occurred_at END), '0000-00-00 00:00:00'), '1970-01-01')
             THEN 1 ELSE 0 END AS liked,
        MAX(CASE WHEN LOWER(event_type) = 'share' THEN 1 ELSE 0 END) AS shared,
        NULLIF(MIN(occurred_at), '0000-00-00 00:00:00') AS first_seen_at,
        NULLIF(MAX(occurred_at), '0000-00-00 00:00:00') AS last_seen_at,
        (
            SUM(CASE WHEN LOWER(event_type) = 'article_in' THEN 1.0 ELSE 0 END) * 1.0 +
            SUM(CASE WHEN LOWER(event_type) = 'like' THEN 2.0
                      WHEN LOWER(event_type) = 'like_cancel' THEN -2.0 ELSE 0 END) +
            SUM(CASE WHEN LOWER(event_type) = 'archive' THEN 2.0
                      WHEN LOWER(event_type) = 'archive_cancel' THEN -2.0 ELSE 0 END) +
            SUM(CASE WHEN LOWER(event_type) = 'share' THEN 1.0 ELSE 0 END) * 2.0 +
            SUM(CASE WHEN LOWER(event_type) = 'f_imp' THEN 1.0 ELSE 0 END) * 0.05 +
            SUM(CASE WHEN LOWER(event_type) = 'uninterest' THEN 1.0 ELSE 0 END) * -3.0
        ) AS engagement_score
    FROM user_events
    WHERE article_id IS NOT NULL
    GROUP BY member_id, article_id
    ON DUPLICATE KEY UPDATE
        impressions = VALUES(impressions),
        clicks = VALUES(clicks),
        total_dwell_ms = VALUES(total_dwell_ms),
        max_scroll_depth = VALUES(max_scroll_depth),
        bookmarked = VALUES(bookmarked),
        liked = VALUES(liked),
        shared = VALUES(shared),
        last_seen_at = VALUES(last_seen_at),
        engagement_score = VALUES(engagement_score)
    """

    # 같은 connection 안에서 SET → SELECT/UPSERT (다른 connection 에 새는 일 없음)
    with conn.engine.connect() as c:
        c.execute(text(f"SET SESSION sql_mode = '{_RELAXED_SQL_MODE}'"))
        c.execute(text(sql))
        c.commit()
    logger.info("user_article_engagement 집계 완료")
