"""
Phase 1: per-(member, category) Beta TS state daily reconcile.

Ground-truth recompute:
  α = prior_α + Σ(positive_event_weight)
  β = prior_β + Σ(negative_event_weight) + impression_no_engagement * w

서빙 측 실시간 update와 race 가능 — 다음 reconcile에서 ground-truth로 정정 (idempotent).

입력 테이블:
  - user_events    (event_type, article_id, member_id, occurred_at)
  - article        (article_id → category_id 매핑)
  - member_interest (onboarding 카테고리)
  - recommendation_impression (impression 후 click 없는 케이스 → β 약한 negative)
"""
from __future__ import annotations

from typing import Dict

import polars as pl
from sqlalchemy import text

from utils.logger import get_logger

logger = get_logger("BanditReconcile")


def _fetch_event_aggregates(conn, lookback_days: int) -> pl.DataFrame:
    """
    유저 × 카테고리 × event_type 별 카운트.
    article.category_id로 카테고리 매핑.
    """
    sql = f"""
    SELECT
        e.member_id                              AS member_id,
        a.category_id                             AS category_id,
        LOWER(e.event_type)                       AS event_type,
        COUNT(*)                                  AS cnt
    FROM user_events e
    JOIN article a
      ON e.article_id = a.article_id
    WHERE e.member_id IS NOT NULL
      AND e.article_id IS NOT NULL
      AND a.category_id IS NOT NULL
      AND e.occurred_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND LOWER(e.event_type) IN ('article_in','like','archive','share','uninterest')
    GROUP BY e.member_id, a.category_id, LOWER(e.event_type)
    """
    return conn.execute(sql)


def _fetch_impression_no_engagement(conn, lookback_days: int, click_window_hours: int) -> pl.DataFrame:
    """
    impression 후 click_window 안에 article_in / like / archive / share 가 없는 노출 수.
    """
    sql = f"""
    SELECT
        i.member_id                              AS member_id,
        i.category_id                             AS category_id,
        COUNT(*)                                  AS impression_no_eng
    FROM recommendation_impression i
    LEFT JOIN user_events e
      ON e.member_id = i.member_id
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) IN ('article_in','like','archive','share')
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL {int(click_window_hours)} HOUR
    WHERE i.shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND i.shown_at <  NOW() - INTERVAL {int(click_window_hours)} HOUR
      AND i.category_id IS NOT NULL
      AND e.id IS NULL
    GROUP BY i.member_id, i.category_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={
            "member_id": pl.Int64, "category_id": pl.Int64, "impression_no_eng": pl.Int64,
        })
    return df


def _fetch_impression_clicks(conn, lookback_days: int) -> pl.DataFrame:
    """member_category_bandit 의 impressions/clicks 컬럼 채울 raw 카운트."""
    sql = f"""
    SELECT
        i.member_id                              AS member_id,
        i.category_id                             AS category_id,
        COUNT(*)                                  AS impressions,
        SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END) AS clicks
    FROM recommendation_impression i
    LEFT JOIN user_events e
      ON e.member_id = i.member_id
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) = 'article_in'
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL 24 HOUR
    WHERE i.shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND i.category_id IS NOT NULL
    GROUP BY i.member_id, i.category_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={
            "member_id": pl.Int64, "category_id": pl.Int64,
            "impressions": pl.Int64, "clicks": pl.Int64,
        })
    return df


def _fetch_member_interest(conn) -> pl.DataFrame:
    sql = """
    SELECT member_id, interest_id AS category_id
    FROM member_interest
    """
    return conn.execute(sql)


def _fetch_active_members(conn) -> pl.DataFrame:
    """
    풀에 있는 카테고리 × 모든 active member 의 cartesian 풀 만들기 위해 active member 조회.
    member 수가 폭증하면 필터링 정책 추가 (status='ACTIVE' 등).
    """
    sql = """
    SELECT member_id
    FROM member
    WHERE status = 'ACTIVE'
      AND role IN ('ROLE_USER', 'ROLE_GUEST')
    """
    return conn.execute(sql)


def _fetch_pool_categories(conn) -> pl.DataFrame:
    sql = "SELECT DISTINCT category_id FROM recommendation_global WHERE category_id IS NOT NULL"
    return conn.execute(sql)


def _fetch_active_devices(conn) -> pl.DataFrame:
    """device_category_bandit 에 row 가 있는 device 만 reconcile (lazy init 된 device).
    아직 활동 안 한 device 까지 cartesian 으로 뽑으면 row 폭증 — bandit 자체가 active 의 의미.
    """
    sql = "SELECT DISTINCT device_id FROM device_category_bandit"
    return conn.execute(sql)


def _fetch_device_event_aggregates(conn, lookback_days: int) -> pl.DataFrame:
    """device × category × event_type 카운트 (member_id NULL + device_id 있는 user_events)."""
    sql = f"""
    SELECT
        e.device_id                              AS device_id,
        a.category_id                             AS category_id,
        LOWER(e.event_type)                       AS event_type,
        COUNT(*)                                  AS cnt
    FROM user_events e
    JOIN article a
      ON e.article_id = a.article_id
    WHERE e.member_id IS NULL
      AND e.device_id IS NOT NULL
      AND e.article_id IS NOT NULL
      AND a.category_id IS NOT NULL
      AND e.occurred_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND LOWER(e.event_type) IN ('article_in','like','archive','share','uninterest')
    GROUP BY e.device_id, a.category_id, LOWER(e.event_type)
    """
    return conn.execute(sql)


def _fetch_device_impression_no_engagement(conn, lookback_days: int, click_window_hours: int) -> pl.DataFrame:
    sql = f"""
    SELECT
        i.device_id                              AS device_id,
        i.category_id                             AS category_id,
        COUNT(*)                                  AS impression_no_eng
    FROM recommendation_impression i
    LEFT JOIN user_events e
      ON e.device_id = i.device_id
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) IN ('article_in','like','archive','share')
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL {int(click_window_hours)} HOUR
    WHERE i.shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND i.shown_at <  NOW() - INTERVAL {int(click_window_hours)} HOUR
      AND i.device_id IS NOT NULL
      AND i.category_id IS NOT NULL
      AND e.id IS NULL
    GROUP BY i.device_id, i.category_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={
            "device_id": pl.Utf8, "category_id": pl.Int64, "impression_no_eng": pl.Int64,
        })
    return df


def _fetch_device_impression_clicks(conn, lookback_days: int) -> pl.DataFrame:
    sql = f"""
    SELECT
        i.device_id                              AS device_id,
        i.category_id                             AS category_id,
        COUNT(*)                                  AS impressions,
        SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END) AS clicks
    FROM recommendation_impression i
    LEFT JOIN user_events e
      ON e.device_id = i.device_id
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) = 'article_in'
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL 24 HOUR
    WHERE i.shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND i.device_id IS NOT NULL
      AND i.category_id IS NOT NULL
    GROUP BY i.device_id, i.category_id
    """
    df = conn.execute(sql)
    if df.is_empty():
        return pl.DataFrame(schema={
            "device_id": pl.Utf8, "category_id": pl.Int64,
            "impressions": pl.Int64, "clicks": pl.Int64,
        })
    return df


def reconcile(conn, config: Dict) -> Dict:
    """
    member_category_bandit ground-truth overwrite.
    Returns metric payload.
    """
    cfg = config.get("bandit", {})
    rollup_cfg = config.get("metric_rollup", {})

    prior_a_on = float(cfg.get("prior_alpha_onboarding", 4.0))
    prior_b_on = float(cfg.get("prior_beta_onboarding", 1.0))
    prior_a_def = float(cfg.get("prior_alpha_default", 1.0))
    prior_b_def = float(cfg.get("prior_beta_default", 2.0))
    rewards = cfg.get("reward", {})
    lookback_days = int(cfg.get("reconcile_lookback_days", 30))
    click_window = int(rollup_cfg.get("click_window_hours", 24))

    members = _fetch_active_members(conn)
    categories = _fetch_pool_categories(conn)

    if members.is_empty() or categories.is_empty():
        logger.warning(f"members={len(members)} categories={len(categories)} — bandit reconcile 생략")
        return {"members": int(len(members)), "categories": int(len(categories)), "rows": 0}

    # cartesian product (member × category) — 모든 카테고리에 prior 부여
    base = members.join(categories, how="cross")

    interests = _fetch_member_interest(conn)
    if not interests.is_empty():
        interests = interests.with_columns(pl.lit(1).alias("is_onboarding"))
        base = base.join(interests, on=["member_id", "category_id"], how="left")
        base = base.with_columns(pl.col("is_onboarding").fill_null(0))
    else:
        base = base.with_columns(pl.lit(0).alias("is_onboarding"))

    # event 신호
    events = _fetch_event_aggregates(conn, lookback_days)
    pos_keys = {"article_in", "like", "archive", "share"}
    neg_keys = {"uninterest"}

    if events.is_empty():
        events_pivot = pl.DataFrame(schema={
            "member_id": pl.Int64, "category_id": pl.Int64,
            "alpha_delta_event": pl.Float64, "beta_delta_event": pl.Float64,
        })
    else:
        events = events.with_columns([
            pl.col("event_type").map_elements(
                lambda et: float(rewards.get(et, 0.0)) if et in pos_keys else 0.0,
                return_dtype=pl.Float64,
            ).alias("alpha_w"),
            pl.col("event_type").map_elements(
                lambda et: -float(rewards.get(et, 0.0)) if et in neg_keys else 0.0,
                return_dtype=pl.Float64,
            ).alias("beta_w"),  # uninterest reward = -2.0 → β += 2
        ])
        events = events.with_columns([
            (pl.col("alpha_w") * pl.col("cnt")).alias("alpha_delta_event"),
            (pl.col("beta_w") * pl.col("cnt")).alias("beta_delta_event"),
        ])
        events_pivot = (
            events.group_by(["member_id", "category_id"])
            .agg([
                pl.col("alpha_delta_event").sum(),
                pl.col("beta_delta_event").sum(),
            ])
        )

    # impression-no-engagement → β
    inoe = _fetch_impression_no_engagement(conn, lookback_days, click_window)
    inoe_w = -float(rewards.get("impression_no_engagement_after_24h", -0.1))  # default 0.1
    if not inoe.is_empty():
        inoe = inoe.with_columns(
            (pl.col("impression_no_eng").cast(pl.Float64) * inoe_w).alias("beta_delta_inoe")
        )
    else:
        inoe = pl.DataFrame(schema={
            "member_id": pl.Int64, "category_id": pl.Int64,
            "impression_no_eng": pl.Int64, "beta_delta_inoe": pl.Float64,
        })

    # impression / click counts (별도 컬럼)
    impressions = _fetch_impression_clicks(conn, lookback_days)

    df = base
    df = df.join(events_pivot, on=["member_id", "category_id"], how="left")
    df = df.join(inoe.select(["member_id", "category_id", "beta_delta_inoe"]),
                 on=["member_id", "category_id"], how="left")
    df = df.join(impressions, on=["member_id", "category_id"], how="left")

    df = df.with_columns([
        pl.col("alpha_delta_event").fill_null(0.0),
        pl.col("beta_delta_event").fill_null(0.0),
        pl.col("beta_delta_inoe").fill_null(0.0),
        pl.col("impressions").fill_null(0).cast(pl.Int64),
        pl.col("clicks").fill_null(0).cast(pl.Int64),
    ])

    df = df.with_columns([
        pl.when(pl.col("is_onboarding") == 1).then(prior_a_on).otherwise(prior_a_def).alias("prior_a"),
        pl.when(pl.col("is_onboarding") == 1).then(prior_b_on).otherwise(prior_b_def).alias("prior_b"),
    ])

    df = df.with_columns([
        (pl.col("prior_a") + pl.col("alpha_delta_event")).alias("alpha"),
        (pl.col("prior_b") + pl.col("beta_delta_event") + pl.col("beta_delta_inoe")).alias("beta"),
    ])

    # 안전: alpha/beta 최소값 0.01 (Beta 0 이면 sampling 폭발)
    df = df.with_columns([
        pl.col("alpha").clip(lower_bound=0.01),
        pl.col("beta").clip(lower_bound=0.01),
    ])

    # UPSERT
    rows = [
        {
            "member_id": int(r["member_id"]),
            "category_id": int(r["category_id"]),
            "alpha": float(r["alpha"]),
            "beta": float(r["beta"]),
            "impressions": int(r["impressions"]),
            "clicks": int(r["clicks"]),
        }
        for r in df.iter_rows(named=True)
    ]

    upsert_sql = """
        INSERT INTO member_category_bandit
            (member_id, category_id, alpha, beta, impressions, clicks)
        VALUES (:member_id, :category_id, :alpha, :beta, :impressions, :clicks)
        ON DUPLICATE KEY UPDATE
            alpha       = VALUES(alpha),
            beta        = VALUES(beta),
            impressions = VALUES(impressions),
            clicks      = VALUES(clicks)
    """
    with conn.engine.connect() as c:
        with c.begin():
            batch_size = 500
            for i in range(0, len(rows), batch_size):
                c.execute(text(upsert_sql), rows[i : i + batch_size])

    payload = {
        "rows": len(rows),
        "members": int(len(members)),
        "categories": int(len(categories)),
        "events_total": int(events["cnt"].sum()) if not events.is_empty() else 0,
        "impressions_total": int(df["impressions"].sum()),
        "clicks_total": int(df["clicks"].sum()),
        "alpha_mean": round(float(df["alpha"].mean()), 4),
        "beta_mean": round(float(df["beta"].mean()), 4),
    }
    logger.info(f"bandit_reconcile done: {payload}")
    return payload


def reconcile_devices(conn, config: Dict) -> Dict:
    """device_category_bandit ground-truth overwrite.

    interest_ids prior 효과는 lazy init 후 며칠 안에 event 가 안 쌓이면 균등으로 회복 — 알려진 trade-off.
    실시간 incremental update 의 안전망 + impression-no-engagement β 보정 책임.
    """
    cfg = config.get("bandit", {})
    rollup_cfg = config.get("metric_rollup", {})

    rewards = cfg.get("reward", {})
    lookback_days = int(cfg.get("reconcile_lookback_days", 30))
    click_window = int(rollup_cfg.get("click_window_hours", 24))
    # device 균등 prior — recsys-serving bandit.py 의 DEVICE_PRIOR_* 와 동기화.
    prior_a = float(cfg.get("device_prior_alpha", 1.0))
    prior_b = float(cfg.get("device_prior_beta", 1.0))

    devices = _fetch_active_devices(conn)
    categories = _fetch_pool_categories(conn)
    if devices.is_empty() or categories.is_empty():
        logger.warning(f"devices={len(devices)} categories={len(categories)} — device reconcile 생략")
        return {"devices": int(len(devices)), "categories": int(len(categories)), "rows": 0}

    base = devices.join(categories, how="cross")

    events = _fetch_device_event_aggregates(conn, lookback_days)
    pos_keys = {"article_in", "like", "archive", "share"}
    neg_keys = {"uninterest"}
    if events.is_empty():
        events_pivot = pl.DataFrame(schema={
            "device_id": pl.Utf8, "category_id": pl.Int64,
            "alpha_delta_event": pl.Float64, "beta_delta_event": pl.Float64,
        })
    else:
        events = events.with_columns([
            pl.col("event_type").map_elements(
                lambda et: float(rewards.get(et, 0.0)) if et in pos_keys else 0.0,
                return_dtype=pl.Float64,
            ).alias("alpha_w"),
            pl.col("event_type").map_elements(
                lambda et: -float(rewards.get(et, 0.0)) if et in neg_keys else 0.0,
                return_dtype=pl.Float64,
            ).alias("beta_w"),
        ])
        events = events.with_columns([
            (pl.col("alpha_w") * pl.col("cnt")).alias("alpha_delta_event"),
            (pl.col("beta_w") * pl.col("cnt")).alias("beta_delta_event"),
        ])
        events_pivot = (
            events.group_by(["device_id", "category_id"])
            .agg([
                pl.col("alpha_delta_event").sum(),
                pl.col("beta_delta_event").sum(),
            ])
        )

    inoe = _fetch_device_impression_no_engagement(conn, lookback_days, click_window)
    inoe_w = -float(rewards.get("impression_no_engagement_after_24h", -0.1))
    if not inoe.is_empty():
        inoe = inoe.with_columns(
            (pl.col("impression_no_eng").cast(pl.Float64) * inoe_w).alias("beta_delta_inoe")
        )
    else:
        inoe = pl.DataFrame(schema={
            "device_id": pl.Utf8, "category_id": pl.Int64,
            "impression_no_eng": pl.Int64, "beta_delta_inoe": pl.Float64,
        })

    impressions = _fetch_device_impression_clicks(conn, lookback_days)

    df = base
    df = df.join(events_pivot, on=["device_id", "category_id"], how="left")
    df = df.join(inoe.select(["device_id", "category_id", "beta_delta_inoe"]),
                 on=["device_id", "category_id"], how="left")
    df = df.join(impressions, on=["device_id", "category_id"], how="left")

    df = df.with_columns([
        pl.col("alpha_delta_event").fill_null(0.0),
        pl.col("beta_delta_event").fill_null(0.0),
        pl.col("beta_delta_inoe").fill_null(0.0),
        pl.col("impressions").fill_null(0).cast(pl.Int64),
        pl.col("clicks").fill_null(0).cast(pl.Int64),
    ])

    df = df.with_columns([
        (pl.lit(prior_a) + pl.col("alpha_delta_event")).alias("alpha"),
        (pl.lit(prior_b) + pl.col("beta_delta_event") + pl.col("beta_delta_inoe")).alias("beta"),
    ])
    df = df.with_columns([
        pl.col("alpha").clip(lower_bound=0.01),
        pl.col("beta").clip(lower_bound=0.01),
    ])

    rows = [
        {
            "device_id": str(r["device_id"]),
            "category_id": int(r["category_id"]),
            "alpha": float(r["alpha"]),
            "beta": float(r["beta"]),
            "impressions": int(r["impressions"]),
            "clicks": int(r["clicks"]),
        }
        for r in df.iter_rows(named=True)
    ]

    upsert_sql = """
        INSERT INTO device_category_bandit
            (device_id, category_id, alpha, beta, impressions, clicks)
        VALUES (:device_id, :category_id, :alpha, :beta, :impressions, :clicks)
        ON DUPLICATE KEY UPDATE
            alpha       = VALUES(alpha),
            beta        = VALUES(beta),
            impressions = VALUES(impressions),
            clicks      = VALUES(clicks)
    """
    with conn.engine.connect() as c:
        with c.begin():
            batch_size = 500
            for i in range(0, len(rows), batch_size):
                c.execute(text(upsert_sql), rows[i : i + batch_size])

    payload = {
        "rows": len(rows),
        "devices": int(len(devices)),
        "categories": int(len(categories)),
        "events_total": int(events["cnt"].sum()) if not events.is_empty() else 0,
        "impressions_total": int(df["impressions"].sum()),
        "clicks_total": int(df["clicks"].sum()),
        "alpha_mean": round(float(df["alpha"].mean()), 4),
        "beta_mean": round(float(df["beta"].mean()), 4),
    }
    logger.info(f"bandit_reconcile_devices done: {payload}")
    return payload
