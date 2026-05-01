"""
Per-(key, category) Beta TS state daily reconcile.

회원 (`member_category_bandit`) 과 비회원 device (`device_category_bandit`) 가 동일 알고리즘.
key 컬럼 (`member_id` / `device_id`) 과 prior 결정만 다름 — 공통 reconcile core 로 통합.

Ground-truth recompute:
  α = prior_α + Σ(positive_event_weight)
  β = prior_β + Σ(negative_event_weight) + impression_no_engagement * w

서빙 측 실시간 update와 race 가능 — 다음 reconcile에서 ground-truth로 정정 (idempotent).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import polars as pl
from sqlalchemy import text

from utils.logger import get_logger

logger = get_logger("BanditReconcile")

POS_KEYS = {"article_in", "like", "archive", "share"}
NEG_KEYS = {"uninterest"}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _fetch_member_interest(conn) -> pl.DataFrame:
    return conn.execute(
        "SELECT mi.member_id, mi.interest_id AS category_id FROM member_interest mi"
    )


def _fetch_pool_categories(conn) -> pl.DataFrame:
    return conn.execute(
        "SELECT DISTINCT category_id FROM recommendation_global WHERE category_id IS NOT NULL"
    )


# ---------------------------------------------------------------------------
# ReconcileSpec — key 컬럼/PK 테이블 + prior 결정 + interest join 만 다름
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReconcileSpec:
    name: str
    key_col: str               # member_id | device_id
    table: str                 # member_category_bandit | device_category_bandit
    pl_key_dtype: pl.DataType  # 빈 schema fallback 시 사용
    fetch_keys: Callable       # (conn, lookback_days) → pl.DataFrame[key_col]
    prior_alpha_default: float
    prior_beta_default: float
    interest_join: bool        # 회원만 onboarding → split prior


# ---------------------------------------------------------------------------
# Fetch helpers — key_col 만 f-string 으로 다름. key_col 은 코드 상수 (injection 안전).
# ---------------------------------------------------------------------------

def _fetch_event_aggregates(conn, key_col: str, lookback_days: int) -> pl.DataFrame:
    """user_events × article 로 카테고리 매핑.
    회원: member_id NOT NULL. 비회원 device: member_id NULL + device_id NOT NULL."""
    member_filter = (
        "e.member_id IS NOT NULL" if key_col == "member_id"
        else "e.member_id IS NULL AND e.device_id IS NOT NULL"
    )
    sql = f"""
    SELECT
        e.{key_col}                              AS {key_col},
        a.category_id                             AS category_id,
        LOWER(e.event_type)                       AS event_type,
        COUNT(*)                                  AS cnt
    FROM user_events e
    JOIN article a
      ON e.article_id = a.article_id
    WHERE {member_filter}
      AND e.article_id IS NOT NULL
      AND a.category_id IS NOT NULL
      AND e.occurred_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND LOWER(e.event_type) IN ('article_in','like','archive','share','uninterest')
    GROUP BY e.{key_col}, a.category_id, LOWER(e.event_type)
    """
    return conn.execute(sql)


def _fetch_impression_no_engagement(
    conn, key_col: str, lookback_days: int, click_window_hours: int
) -> pl.DataFrame:
    win = int(click_window_hours)
    sql = f"""
    SELECT
        i.{key_col}                              AS {key_col},
        i.category_id                             AS category_id,
        COUNT(*)                                  AS impression_no_eng
    FROM recommendation_impression i
    LEFT JOIN user_events e
      ON e.{key_col} = i.{key_col}
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) IN ('article_in','like','archive','share')
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL {win} HOUR
    WHERE i.shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND i.shown_at <  NOW() - INTERVAL {win} HOUR
      AND i.{key_col} IS NOT NULL
      AND i.category_id IS NOT NULL
      AND e.id IS NULL
    GROUP BY i.{key_col}, i.category_id
    """
    return conn.execute(sql)


def _fetch_impression_clicks(conn, key_col: str, lookback_days: int) -> pl.DataFrame:
    sql = f"""
    SELECT
        i.{key_col}                              AS {key_col},
        i.category_id                             AS category_id,
        COUNT(*)                                  AS impressions,
        SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END) AS clicks
    FROM recommendation_impression i
    LEFT JOIN user_events e
      ON e.{key_col} = i.{key_col}
     AND e.article_id = i.article_id
     AND LOWER(e.event_type) = 'article_in'
     AND e.occurred_at >= i.shown_at
     AND e.occurred_at <  i.shown_at + INTERVAL 24 HOUR
    WHERE i.shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
      AND i.{key_col} IS NOT NULL
      AND i.category_id IS NOT NULL
    GROUP BY i.{key_col}, i.category_id
    """
    return conn.execute(sql)


def _fetch_active_members(conn, _lookback_days: int) -> pl.DataFrame:
    return conn.execute(
        """
        SELECT member_id FROM member
         WHERE status = 'ACTIVE' AND role IN ('ROLE_USER', 'ROLE_GUEST')
        """
    )


def _fetch_active_devices(conn, lookback_days: int) -> pl.DataFrame:
    """lookback 안 활동 (user_events 또는 impression) device 만. dead device 제거."""
    sql = f"""
    SELECT DISTINCT device_id FROM (
        SELECT device_id FROM user_events
         WHERE device_id IS NOT NULL
           AND occurred_at >= NOW() - INTERVAL {int(lookback_days)} DAY
        UNION
        SELECT device_id FROM recommendation_impression
         WHERE device_id IS NOT NULL
           AND shown_at >= NOW() - INTERVAL {int(lookback_days)} DAY
    ) t
    """
    return conn.execute(sql)


# ---------------------------------------------------------------------------
# Generic reconcile core
# ---------------------------------------------------------------------------

def _reconcile_core(conn, config: Dict, spec: ReconcileSpec) -> Dict:
    cfg = config.get("bandit", {})
    rollup_cfg = config.get("metric_rollup", {})
    rewards = cfg.get("reward", {})
    lookback_days = int(cfg.get("reconcile_lookback_days", 30))
    click_window = int(rollup_cfg.get("click_window_hours", 24))
    prior_a_def = spec.prior_alpha_default
    prior_b_def = spec.prior_beta_default
    prior_a_on = float(cfg.get("prior_alpha_onboarding", 4.0))
    prior_b_on = float(cfg.get("prior_beta_onboarding", 1.0))
    inoe_w = -float(rewards.get("impression_no_engagement_after_24h", -0.1))

    keys = spec.fetch_keys(conn, lookback_days)
    categories = _fetch_pool_categories(conn)
    if keys.is_empty() or categories.is_empty():
        logger.warning(f"{spec.name}: keys={len(keys)} categories={len(categories)} — 생략")
        return {"name": spec.name, "rows": 0, "keys": int(len(keys)), "categories": int(len(categories))}

    base = keys.join(categories, how="cross")

    if spec.interest_join:
        interests = _fetch_member_interest(conn)
        if not interests.is_empty():
            interests = interests.with_columns(pl.lit(1).alias("is_onboarding"))
            base = base.join(interests, on=[spec.key_col, "category_id"], how="left")
            base = base.with_columns(pl.col("is_onboarding").fill_null(0))
        else:
            base = base.with_columns(pl.lit(0).alias("is_onboarding"))
    else:
        base = base.with_columns(pl.lit(0).alias("is_onboarding"))

    events = _fetch_event_aggregates(conn, spec.key_col, lookback_days)
    if events.is_empty():
        events_pivot = pl.DataFrame(schema={
            spec.key_col: spec.pl_key_dtype, "category_id": pl.Int64,
            "alpha_delta_event": pl.Float64, "beta_delta_event": pl.Float64,
        })
    else:
        events = events.with_columns([
            pl.col("event_type").map_elements(
                lambda et: float(rewards.get(et, 0.0)) if et in POS_KEYS else 0.0,
                return_dtype=pl.Float64,
            ).alias("alpha_w"),
            pl.col("event_type").map_elements(
                lambda et: -float(rewards.get(et, 0.0)) if et in NEG_KEYS else 0.0,
                return_dtype=pl.Float64,
            ).alias("beta_w"),
        ])
        events = events.with_columns([
            (pl.col("alpha_w") * pl.col("cnt")).alias("alpha_delta_event"),
            (pl.col("beta_w") * pl.col("cnt")).alias("beta_delta_event"),
        ])
        events_pivot = (
            events.group_by([spec.key_col, "category_id"]).agg([
                pl.col("alpha_delta_event").sum(),
                pl.col("beta_delta_event").sum(),
            ])
        )

    inoe = _fetch_impression_no_engagement(conn, spec.key_col, lookback_days, click_window)
    if inoe.is_empty():
        inoe = pl.DataFrame(schema={
            spec.key_col: spec.pl_key_dtype, "category_id": pl.Int64,
            "impression_no_eng": pl.Int64, "beta_delta_inoe": pl.Float64,
        })
    else:
        inoe = inoe.with_columns(
            (pl.col("impression_no_eng").cast(pl.Float64) * inoe_w).alias("beta_delta_inoe")
        )

    impressions = _fetch_impression_clicks(conn, spec.key_col, lookback_days)
    if impressions.is_empty():
        impressions = pl.DataFrame(schema={
            spec.key_col: spec.pl_key_dtype, "category_id": pl.Int64,
            "impressions": pl.Int64, "clicks": pl.Int64,
        })

    df = base
    df = df.join(events_pivot, on=[spec.key_col, "category_id"], how="left")
    df = df.join(inoe.select([spec.key_col, "category_id", "beta_delta_inoe"]),
                 on=[spec.key_col, "category_id"], how="left")
    df = df.join(impressions, on=[spec.key_col, "category_id"], how="left")

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
    df = df.with_columns([
        pl.col("alpha").clip(lower_bound=0.01),
        pl.col("beta").clip(lower_bound=0.01),
    ])

    rows = [
        {
            spec.key_col: r[spec.key_col] if spec.key_col == "device_id" else int(r[spec.key_col]),
            "category_id": int(r["category_id"]),
            "alpha": float(r["alpha"]),
            "beta": float(r["beta"]),
            "impressions": int(r["impressions"]),
            "clicks": int(r["clicks"]),
        }
        for r in df.iter_rows(named=True)
    ]

    upsert_sql = f"""
        INSERT INTO {spec.table}
            ({spec.key_col}, category_id, alpha, beta, impressions, clicks)
        VALUES (:{spec.key_col}, :category_id, :alpha, :beta, :impressions, :clicks)
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
        "name": spec.name,
        "rows": len(rows),
        "keys": int(len(keys)),
        "categories": int(len(categories)),
        "events_total": int(events["cnt"].sum()) if not events.is_empty() else 0,
        "impressions_total": int(df["impressions"].sum()),
        "clicks_total": int(df["clicks"].sum()),
        "alpha_mean": round(float(df["alpha"].mean()), 4),
        "beta_mean": round(float(df["beta"].mean()), 4),
    }
    logger.info(f"{spec.name} done: {payload}")
    return payload


# ---------------------------------------------------------------------------
# Public entry points — main.py 가 호출
# ---------------------------------------------------------------------------

_MEMBER_SPEC = ReconcileSpec(
    name="bandit_reconcile",
    key_col="member_id",
    table="member_category_bandit",
    pl_key_dtype=pl.Int64,
    fetch_keys=_fetch_active_members,
    prior_alpha_default=1.0,
    prior_beta_default=2.0,
    interest_join=True,
)

_DEVICE_SPEC = ReconcileSpec(
    name="bandit_reconcile_devices",
    key_col="device_id",
    table="device_category_bandit",
    pl_key_dtype=pl.Utf8,
    fetch_keys=_fetch_active_devices,
    prior_alpha_default=1.0,
    prior_beta_default=1.0,
    interest_join=False,
)


def reconcile(conn, config: Dict) -> Dict:
    """member_category_bandit ground-truth overwrite."""
    return _reconcile_core(conn, config, _MEMBER_SPEC)


def reconcile_devices(conn, config: Dict) -> Dict:
    """device_category_bandit ground-truth overwrite (lookback active filter).

    interest_ids prior 효과는 lazy init 후 며칠 안 event 가 안 쌓이면 균등으로 회복 — 알려진 trade-off.
    """
    return _reconcile_core(conn, config, _DEVICE_SPEC)
