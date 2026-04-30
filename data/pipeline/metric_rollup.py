"""
일별 추천 KPI rollup.

테이블:
  - recommendation_metric_daily : 어제 기준 CTR, per-category, per-position, diversity, freshness
  - bandit_state_snapshot       : 모든 (member, category) bandit 상태 스냅샷
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Dict, List

import polars as pl
from sqlalchemy import text

from utils.logger import get_logger

logger = get_logger("MetricRollup")


def _yesterday_range(now: dt.datetime) -> tuple[dt.datetime, dt.datetime, dt.date]:
    today = now.date()
    yesterday = today - dt.timedelta(days=1)
    start = dt.datetime.combine(yesterday, dt.time.min)
    end = dt.datetime.combine(today, dt.time.min)
    return start, end, yesterday


def _fetch_impressions_window(conn, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
    sql = """
    SELECT
        impression_id,
        member_id,
        CAST(article_id AS CHAR) AS article_id,
        category_id,
        position,
        bandit_theta,
        shown_at
    FROM recommendation_impression
    WHERE shown_at >= :start AND shown_at < :end
    """
    with conn.engine.connect() as c:
        df = pl.read_database(text(sql).bindparams(start=start, end=end), c)
    return df


def _fetch_clicks_for_impressions(
    conn,
    start: dt.datetime,
    end: dt.datetime,
    click_window_hours: int,
) -> pl.DataFrame:
    """impression 후 click_window 안 article_in click 만 reward 인정."""
    win = int(click_window_hours)
    sql = f"""
    SELECT
        i.impression_id        AS impression_id,
        MIN(e.occurred_at)      AS click_at
    FROM recommendation_impression i
    JOIN user_events e
      ON e.member_id = i.member_id
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) = 'article_in'
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL {win} HOUR
    WHERE i.shown_at >= :start AND i.shown_at < :end
    GROUP BY i.impression_id
    """
    with conn.engine.connect() as c:
        df = pl.read_database(text(sql).bindparams(start=start, end=end), c)
    return df


def _fetch_article_freshness(conn, article_ids: List[str]) -> pl.DataFrame:
    if not article_ids:
        return pl.DataFrame(schema={"article_id": pl.Utf8, "days_old": pl.Float64})

    from sqlalchemy import bindparam

    sql = """
    SELECT
        CAST(article_id AS CHAR) AS article_id,
        TIMESTAMPDIFF(HOUR, published_at, NOW()) / 24.0 AS days_old
    FROM article
    WHERE article_id IN :ids
    """
    stmt = text(sql).bindparams(bindparam("ids", expanding=True))
    with conn.engine.connect() as c:
        df = pl.read_database(stmt.bindparams(ids=article_ids), c)
    return df


def _fetch_member_interest_categories(conn) -> pl.DataFrame:
    sql = "SELECT DISTINCT member_id, interest_id AS category_id FROM member_interest"
    return conn.execute(sql)


def _fetch_pool_size(conn) -> int:
    df = conn.execute("SELECT COUNT(*) AS n FROM recommendation_global")
    if df.is_empty():
        return 0
    return int(df["n"][0])


def _fetch_bandit_active_users(conn) -> int:
    df = conn.execute("SELECT COUNT(DISTINCT member_id) AS n FROM member_category_bandit")
    if df.is_empty():
        return 0
    return int(df["n"][0])


def _diversity_entropy(impressions: pl.DataFrame) -> float:
    """응답 단위 엔트로피의 평균. 응답 = (member_id, shown_at 5초 bucket)."""
    if impressions.is_empty():
        return 0.0

    df = impressions.with_columns([
        (pl.col("shown_at").cast(pl.Datetime).dt.truncate("5s")).alias("response_bucket"),
    ])
    grouped = df.group_by(["member_id", "response_bucket"])
    entropies = []
    for _, g in grouped:
        if len(g) <= 1:
            continue
        cat_counts = g.group_by("category_id").len()
        total = float(cat_counts["len"].sum())
        if total <= 0:
            continue
        probs = (cat_counts["len"].cast(pl.Float64) / total).to_list()
        h = -sum(p * math.log(p, 2) for p in probs if p > 0)
        entropies.append(h)
    if not entropies:
        return 0.0
    return round(sum(entropies) / len(entropies), 4)


def _per_position_ctr(joined: pl.DataFrame) -> List[float]:
    """position 1..10 각각의 CTR. 데이터 없는 position은 None 대신 0.0."""
    if joined.is_empty():
        return [0.0] * 10
    by_pos = (
        joined.group_by("position")
        .agg([
            pl.col("position").count().alias("imp"),
            pl.col("clicked").sum().alias("clk"),
        ])
        .sort("position")
    )
    out = [0.0] * 10
    for r in by_pos.iter_rows(named=True):
        pos = int(r["position"])
        if 1 <= pos <= 10:
            imp = float(r["imp"])
            clk = float(r["clk"])
            out[pos - 1] = round(clk / imp, 4) if imp > 0 else 0.0
    return out


def _per_category(joined: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    if joined.is_empty():
        return {}
    by_cat = (
        joined.group_by("category_id")
        .agg([
            pl.col("category_id").count().alias("imp"),
            pl.col("clicked").sum().alias("clk"),
        ])
    )
    out: Dict[str, Dict[str, float]] = {}
    for r in by_cat.iter_rows(named=True):
        cid = r["category_id"]
        if cid is None:
            continue
        imp = int(r["imp"])
        clk = int(r["clk"])
        ctr = round(clk / imp, 4) if imp > 0 else 0.0
        out[str(int(cid))] = {"impressions": imp, "clicks": clk, "ctr": ctr}
    return out


def _cold_to_warm_users(conn, day: dt.date) -> int:
    """오늘(day) 처음으로 article_in 한 유저 수."""
    sql = """
    SELECT COUNT(DISTINCT member_id) AS n
    FROM (
        SELECT member_id, MIN(DATE(occurred_at)) AS first_click_date
        FROM user_events
        WHERE LOWER(event_type) = 'article_in'
          AND member_id IS NOT NULL
        GROUP BY member_id
    ) t
    WHERE first_click_date = :day
    """
    with conn.engine.connect() as c:
        df = pl.read_database(text(sql).bindparams(day=day), c)
    if df.is_empty():
        return 0
    return int(df["n"][0])


def _snapshot_bandit_state(conn, snapshot_date: dt.date) -> int:
    """member_category_bandit → bandit_state_snapshot (date PK 추가) idempotent."""
    upsert_sql = """
    INSERT INTO bandit_state_snapshot
        (snapshot_date, member_id, category_id, alpha, beta, impressions, clicks)
    SELECT
        :d, member_id, category_id, alpha, beta, impressions, clicks
    FROM member_category_bandit
    ON DUPLICATE KEY UPDATE
        alpha = VALUES(alpha),
        beta = VALUES(beta),
        impressions = VALUES(impressions),
        clicks = VALUES(clicks)
    """
    with conn.engine.connect() as c:
        with c.begin():
            c.execute(text(upsert_sql), {"d": snapshot_date})

    df = conn.execute(
        f"SELECT COUNT(*) AS n FROM bandit_state_snapshot WHERE snapshot_date = '{snapshot_date.isoformat()}'"
    )
    return int(df["n"][0]) if not df.is_empty() else 0


def rollup(conn, config: Dict) -> Dict:
    """
    어제 기준 KPI rollup.
    """
    cfg = config.get("metric_rollup", {})
    click_window = int(cfg.get("click_window_hours", 24))

    now = dt.datetime.now()
    start, end, yday = _yesterday_range(now)

    impressions = _fetch_impressions_window(conn, start, end)
    pool_size = _fetch_pool_size(conn)
    bandit_active = _fetch_bandit_active_users(conn)
    cold_to_warm = _cold_to_warm_users(conn, yday)
    snapshot_rows = _snapshot_bandit_state(conn, yday)

    payload: Dict = {
        "metric_date": yday.isoformat(),
        "impressions_total": int(len(impressions)),
        "pool_size": pool_size,
        "bandit_active_users": bandit_active,
        "cold_to_warm_users": cold_to_warm,
        "bandit_snapshot_rows": snapshot_rows,
    }

    if impressions.is_empty():
        upsert_sql = """
        INSERT INTO recommendation_metric_daily
            (metric_date, impressions, clicks, ctr, onboarding_ctr, non_onboarding_ctr,
             per_category, per_position_ctr, diversity_entropy, freshness_median_days,
             bandit_active_users, pool_size, cold_to_warm_users)
        VALUES (:d, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, :ba, :ps, :c2w)
        ON DUPLICATE KEY UPDATE
            impressions = 0, clicks = 0,
            ctr = NULL, onboarding_ctr = NULL, non_onboarding_ctr = NULL,
            per_category = NULL, per_position_ctr = NULL,
            diversity_entropy = NULL, freshness_median_days = NULL,
            bandit_active_users = :ba, pool_size = :ps, cold_to_warm_users = :c2w
        """
        with conn.engine.connect() as c:
            with c.begin():
                c.execute(text(upsert_sql), {
                    "d": yday, "ba": bandit_active, "ps": pool_size, "c2w": cold_to_warm,
                })
        logger.info(f"metric_rollup: 어제 impression 0건. payload={payload}")
        return payload

    # click join
    clicks = _fetch_clicks_for_impressions(conn, start, end, click_window)
    if clicks.is_empty():
        joined = impressions.with_columns(pl.lit(0).alias("clicked"))
    else:
        clicked_ids = set(clicks["impression_id"].to_list())
        joined = impressions.with_columns(
            pl.col("impression_id").is_in(list(clicked_ids)).cast(pl.Int64).alias("clicked")
        )

    total_imp = int(len(joined))
    total_clk = int(joined["clicked"].sum())
    ctr = round(total_clk / total_imp, 4) if total_imp > 0 else None

    # onboarding vs non-onboarding split
    interests = _fetch_member_interest_categories(conn)
    if not interests.is_empty():
        interests = interests.with_columns(pl.lit(1).alias("is_onboarding"))
        joined2 = joined.join(interests, on=["member_id", "category_id"], how="left")
        joined2 = joined2.with_columns(pl.col("is_onboarding").fill_null(0))
        on_df = joined2.filter(pl.col("is_onboarding") == 1)
        off_df = joined2.filter(pl.col("is_onboarding") == 0)
        onb_ctr = round(float(on_df["clicked"].sum()) / max(1, len(on_df)), 4) if len(on_df) > 0 else None
        non_ctr = round(float(off_df["clicked"].sum()) / max(1, len(off_df)), 4) if len(off_df) > 0 else None
    else:
        onb_ctr = None
        non_ctr = None

    per_cat = _per_category(joined)
    per_pos = _per_position_ctr(joined)
    diversity = _diversity_entropy(joined)

    # freshness median
    article_ids = joined["article_id"].unique().to_list()
    fresh = _fetch_article_freshness(conn, article_ids)
    if not fresh.is_empty():
        fresh_join = joined.join(fresh, on="article_id", how="left")
        median_days = float(fresh_join["days_old"].median() or 0.0)
        median_days = round(median_days, 2)
    else:
        median_days = None

    upsert_sql = """
    INSERT INTO recommendation_metric_daily
        (metric_date, impressions, clicks, ctr, onboarding_ctr, non_onboarding_ctr,
         per_category, per_position_ctr, diversity_entropy, freshness_median_days,
         bandit_active_users, pool_size, cold_to_warm_users)
    VALUES (:d, :imp, :clk, :ctr, :on_ctr, :non_ctr,
            :per_cat, :per_pos, :div, :fresh, :ba, :ps, :c2w)
    ON DUPLICATE KEY UPDATE
        impressions = VALUES(impressions),
        clicks = VALUES(clicks),
        ctr = VALUES(ctr),
        onboarding_ctr = VALUES(onboarding_ctr),
        non_onboarding_ctr = VALUES(non_onboarding_ctr),
        per_category = VALUES(per_category),
        per_position_ctr = VALUES(per_position_ctr),
        diversity_entropy = VALUES(diversity_entropy),
        freshness_median_days = VALUES(freshness_median_days),
        bandit_active_users = VALUES(bandit_active_users),
        pool_size = VALUES(pool_size),
        cold_to_warm_users = VALUES(cold_to_warm_users)
    """
    with conn.engine.connect() as c:
        with c.begin():
            c.execute(text(upsert_sql), {
                "d": yday,
                "imp": total_imp,
                "clk": total_clk,
                "ctr": ctr,
                "on_ctr": onb_ctr,
                "non_ctr": non_ctr,
                "per_cat": json.dumps(per_cat, ensure_ascii=False),
                "per_pos": json.dumps(per_pos),
                "div": diversity,
                "fresh": median_days,
                "ba": bandit_active,
                "ps": pool_size,
                "c2w": cold_to_warm,
            })

    payload.update({
        "impressions": total_imp,
        "clicks": total_clk,
        "ctr": ctr,
        "onboarding_ctr": onb_ctr,
        "non_onboarding_ctr": non_ctr,
        "diversity_entropy": diversity,
        "freshness_median_days": median_days,
        "categories_in_response": len(per_cat),
    })
    logger.info(f"metric_rollup done: {payload}")
    return payload
