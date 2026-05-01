"""
일별 추천 KPI rollup.

테이블:
  - recommendation_metric_daily : 어제 KPI (CTR, per-category, per-position, diversity, freshness,
                                  anonymous 분리, backfill_ratio)
  - bandit_state_snapshot       : (member, category) bandit α/β 일별 스냅샷

TZ: 모든 일자 계산은 Asia/Seoul (MySQL default_time_zone='+09:00' 와 정합).
diversity_entropy 는 5초 bucket 휴리스틱 대신 feed_request_id 로 정확 그룹핑.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Dict, List
from zoneinfo import ZoneInfo

import polars as pl
from sqlalchemy import text

from utils.logger import get_logger

logger = get_logger("MetricRollup")
KST = ZoneInfo("Asia/Seoul")


def _yesterday_range(now_kst: dt.datetime) -> tuple[dt.datetime, dt.datetime, dt.date]:
    today = now_kst.date()
    yesterday = today - dt.timedelta(days=1)
    start = dt.datetime.combine(yesterday, dt.time.min)  # naive — MySQL TZ +09:00 와 일치
    end = dt.datetime.combine(today, dt.time.min)
    return start, end, yesterday


def _fetch_impressions_window(conn, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
    sql = """
    SELECT
        impression_id,
        member_id,
        device_id,
        CAST(article_id AS CHAR) AS article_id,
        category_id,
        position,
        feed_request_id,
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
    """impression 후 click_window 안 article_in click. member_id 또는 device_id 매칭."""
    win = int(click_window_hours)
    sql = f"""
    SELECT
        i.impression_id        AS impression_id,
        MIN(e.occurred_at)      AS click_at
    FROM recommendation_impression i
    JOIN user_events e
      ON (
            (i.member_id IS NOT NULL AND e.member_id = i.member_id)
         OR (i.device_id IS NOT NULL AND e.device_id = i.device_id)
         )
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
    """응답 단위 엔트로피의 평균.

    응답 그룹 키:
      - feed_request_id 있으면 그것 (정확)
      - 없으면 (member_id|device_id, shown_at 5초 bucket) — 구버전 호환 휴리스틱
    """
    if impressions.is_empty():
        return 0.0

    has_fr = "feed_request_id" in impressions.columns and impressions["feed_request_id"].null_count() < len(impressions)
    if has_fr:
        df = impressions.with_columns(
            pl.col("feed_request_id").fill_null("").alias("response_bucket")
        ).filter(pl.col("response_bucket") != "")
    else:
        df = impressions.with_columns([
            (pl.col("shown_at").cast(pl.Datetime).dt.truncate("5s")).alias("response_bucket"),
        ])
    # member_id NULL 인 비회원 row 도 묶기 위해 device_id 도 키에 포함.
    grouped = df.group_by(["member_id", "device_id", "response_bucket"])
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


def _fetch_anonymous_active_devices(conn) -> int:
    df = conn.execute("SELECT COUNT(DISTINCT device_id) AS n FROM device_category_bandit")
    if df.is_empty():
        return 0
    return int(df["n"][0])


def rollup(conn, config: Dict) -> Dict:
    """어제(KST) 기준 KPI rollup. 회원 + 비회원 분리."""
    cfg = config.get("metric_rollup", {})
    click_window = int(cfg.get("click_window_hours", 24))

    now_kst = dt.datetime.now(KST)
    start, end, yday = _yesterday_range(now_kst)

    impressions = _fetch_impressions_window(conn, start, end)
    pool_size = _fetch_pool_size(conn)
    bandit_active = _fetch_bandit_active_users(conn)
    anon_devices = _fetch_anonymous_active_devices(conn)
    cold_to_warm = _cold_to_warm_users(conn, yday)
    snapshot_rows = _snapshot_bandit_state(conn, yday)

    payload: Dict = {
        "metric_date": yday.isoformat(),
        "impressions_total": int(len(impressions)),
        "pool_size": pool_size,
        "bandit_active_users": bandit_active,
        "anonymous_active_devices": anon_devices,
        "cold_to_warm_users": cold_to_warm,
        "bandit_snapshot_rows": snapshot_rows,
    }

    if impressions.is_empty():
        upsert_sql = """
        INSERT INTO recommendation_metric_daily
            (metric_date, impressions, anonymous_impressions, clicks, anonymous_clicks,
             ctr, anonymous_ctr, onboarding_ctr, non_onboarding_ctr,
             per_category, per_position_ctr, diversity_entropy, freshness_median_days,
             bandit_active_users, anonymous_active_devices, pool_size,
             cold_to_warm_users, backfill_ratio)
        VALUES (:d, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                :ba, :ad, :ps, :c2w, NULL)
        ON DUPLICATE KEY UPDATE
            impressions = 0, anonymous_impressions = 0, clicks = 0, anonymous_clicks = 0,
            ctr = NULL, anonymous_ctr = NULL, onboarding_ctr = NULL, non_onboarding_ctr = NULL,
            per_category = NULL, per_position_ctr = NULL,
            diversity_entropy = NULL, freshness_median_days = NULL,
            bandit_active_users = :ba, anonymous_active_devices = :ad,
            pool_size = :ps, cold_to_warm_users = :c2w, backfill_ratio = NULL
        """
        with conn.engine.connect() as c:
            with c.begin():
                c.execute(text(upsert_sql), {
                    "d": yday, "ba": bandit_active, "ad": anon_devices,
                    "ps": pool_size, "c2w": cold_to_warm,
                })
        logger.info(f"metric_rollup: 어제 impression 0건. payload={payload}")
        return payload

    # click join (회원/비회원 모두)
    clicks = _fetch_clicks_for_impressions(conn, start, end, click_window)
    if clicks.is_empty():
        joined = impressions.with_columns(pl.lit(0).alias("clicked"))
    else:
        clicked_ids = set(clicks["impression_id"].to_list())
        joined = impressions.with_columns(
            pl.col("impression_id").is_in(list(clicked_ids)).cast(pl.Int64).alias("clicked")
        )

    # 회원/비회원 분리
    is_anon = pl.col("member_id").is_null()
    total_imp = int(len(joined))
    total_clk = int(joined["clicked"].sum())
    ctr = round(total_clk / total_imp, 4) if total_imp > 0 else None

    anon_df = joined.filter(is_anon)
    anon_imp = int(len(anon_df))
    anon_clk = int(anon_df["clicked"].sum()) if anon_imp else 0
    anon_ctr_v = round(anon_clk / anon_imp, 4) if anon_imp > 0 else None

    # backfill_ratio: 같은 feed_request_id 안에서 카테고리 다양성이 1인 응답 비율 (heuristic).
    # 더 정확히는 service.py 가 backfill 발생 시 표시해서 적재해야 하지만, 그건 별도 metric.
    backfill_ratio_v: float | None = None

    # onboarding vs non-onboarding split — 회원만 의미.
    member_df = joined.filter(~is_anon)
    interests = _fetch_member_interest_categories(conn)
    if not interests.is_empty() and len(member_df) > 0:
        interests = interests.with_columns(pl.lit(1).alias("is_onboarding"))
        joined2 = member_df.join(interests, on=["member_id", "category_id"], how="left")
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
        (metric_date, impressions, anonymous_impressions, clicks, anonymous_clicks,
         ctr, anonymous_ctr, onboarding_ctr, non_onboarding_ctr,
         per_category, per_position_ctr, diversity_entropy, freshness_median_days,
         bandit_active_users, anonymous_active_devices, pool_size,
         cold_to_warm_users, backfill_ratio)
    VALUES (:d, :imp, :ai, :clk, :ac, :ctr, :a_ctr, :on_ctr, :non_ctr,
            :per_cat, :per_pos, :div, :fresh, :ba, :ad, :ps, :c2w, :bf)
    ON DUPLICATE KEY UPDATE
        impressions = VALUES(impressions),
        anonymous_impressions = VALUES(anonymous_impressions),
        clicks = VALUES(clicks),
        anonymous_clicks = VALUES(anonymous_clicks),
        ctr = VALUES(ctr),
        anonymous_ctr = VALUES(anonymous_ctr),
        onboarding_ctr = VALUES(onboarding_ctr),
        non_onboarding_ctr = VALUES(non_onboarding_ctr),
        per_category = VALUES(per_category),
        per_position_ctr = VALUES(per_position_ctr),
        diversity_entropy = VALUES(diversity_entropy),
        freshness_median_days = VALUES(freshness_median_days),
        bandit_active_users = VALUES(bandit_active_users),
        anonymous_active_devices = VALUES(anonymous_active_devices),
        pool_size = VALUES(pool_size),
        cold_to_warm_users = VALUES(cold_to_warm_users),
        backfill_ratio = VALUES(backfill_ratio)
    """
    with conn.engine.connect() as c:
        with c.begin():
            c.execute(text(upsert_sql), {
                "d": yday,
                "imp": total_imp,
                "ai": anon_imp,
                "clk": total_clk,
                "ac": anon_clk,
                "ctr": ctr,
                "a_ctr": anon_ctr_v,
                "on_ctr": onb_ctr,
                "non_ctr": non_ctr,
                "per_cat": json.dumps(per_cat, ensure_ascii=False),
                "per_pos": json.dumps(per_pos),
                "div": diversity,
                "fresh": median_days,
                "ba": bandit_active,
                "ad": anon_devices,
                "ps": pool_size,
                "c2w": cold_to_warm,
                "bf": backfill_ratio_v,
            })

    payload.update({
        "impressions": total_imp,
        "anonymous_impressions": anon_imp,
        "clicks": total_clk,
        "anonymous_clicks": anon_clk,
        "ctr": ctr,
        "anonymous_ctr": anon_ctr_v,
        "onboarding_ctr": onb_ctr,
        "non_onboarding_ctr": non_ctr,
        "diversity_entropy": diversity,
        "freshness_median_days": median_days,
        "categories_in_response": len(per_cat),
    })
    logger.info(f"metric_rollup done: {payload}")
    return payload
