from utils.logger import get_logger

logger = get_logger("EngagementAggregator")


def aggregate_engagement(conn) -> None:
    """
    user_events 테이블에서 유저×아티클별 engagement를 집계하여
    user_article_engagement 테이블에 UPSERT.
    배치 파이프라인의 마지막 단계에서 호출.
    """

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
        CASE WHEN MAX(CASE WHEN LOWER(event_type) = 'archive' THEN occurred_at END)
                  > COALESCE(MAX(CASE WHEN LOWER(event_type) = 'archive_cancel' THEN occurred_at END), '1970-01-01')
             THEN 1 ELSE 0 END AS bookmarked,
        CASE WHEN MAX(CASE WHEN LOWER(event_type) = 'like' THEN occurred_at END)
                  > COALESCE(MAX(CASE WHEN LOWER(event_type) = 'like_cancel' THEN occurred_at END), '1970-01-01')
             THEN 1 ELSE 0 END AS liked,
        MAX(CASE WHEN LOWER(event_type) = 'share' THEN 1 ELSE 0 END) AS shared,
        MIN(occurred_at) AS first_seen_at,
        MAX(occurred_at) AS last_seen_at,
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

    conn._raw_execute(sql)
    logger.info("user_article_engagement 집계 완료")
