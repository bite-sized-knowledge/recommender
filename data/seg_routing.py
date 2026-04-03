import time
import polars as pl
from data.utils import _query_all_active_users


class UserSegmentation:
    """
    MySQL user_events 기반 유저 세그먼트 분류 (cold / warm / hot)
    - 최근 N일 로그 조회 → Polars df
    - 분류 기준: Recency(최근 1/3/7일), 이벤트 수 Quantile
    """

    def __init__(self, conn=None):
        if conn is None:
            raise ValueError("conn을 반드시 제공해야 합니다.")
        self.conn = conn

    # -------------------- Public API --------------------

    def fetch_events_last_ndays(self, days: int = 7) -> pl.DataFrame:
        """최근 N일 이벤트를 MySQL에서 조회하여 Polars DataFrame 반환"""
        sql = f"""
        SELECT
            member_id,
            LOWER(event_type) AS event_type,
            CAST(article_id AS CHAR) AS target_id,
            UNIX_TIMESTAMP(occurred_at) * 1000 AS timestamp
        FROM user_events
        WHERE occurred_at >= NOW() - INTERVAL {days} DAY
          AND article_id IS NOT NULL
        """

        df = self.conn.execute(sql)

        if df.is_empty():
            return pl.DataFrame(
                schema={
                    "member_id": pl.Int64,
                    "event_type": pl.Utf8,
                    "target_id": pl.Utf8,
                    "timestamp": pl.Int64,
                    "datetime_utc": pl.Datetime("ms"),
                }
            )

        df = df.with_columns(
            pl.col("member_id").cast(pl.Int64, strict=False),
            pl.col("timestamp").cast(pl.Int64, strict=False),
        )

        # ms → Datetime[ms, UTC]
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), "ms")
              .dt.replace_time_zone("UTC")
              .alias("datetime_utc")
        )

        return df

    def classify_users_quantile(self, df: pl.DataFrame) -> pl.DataFrame:
        now_ms = int(time.time() * 1000)

        # 유저별 통계
        user_stats = (
            df.group_by("member_id")
              .agg([
                  pl.max("timestamp").alias("last_event_ts"),
                  pl.len().alias("event_count"),
              ])
              .with_columns([
                  ((now_ms - pl.col("last_event_ts")) / (1000 * 60 * 60 * 24)).alias("recency_days")
              ])
        )

        # 분위수 계산
        q1, q2, q3 = (
            user_stats["event_count"].quantile(0.25),
            user_stats["event_count"].quantile(0.50),
            user_stats["event_count"].quantile(0.75),
        )

        # 분류 함수
        def classify(row):
            if row["recency_days"] <= 1 and row["event_count"] >= q3:
                return "hot"
            elif row["recency_days"] <= 3 and row["event_count"] >= q2:
                return "warm"
            else:
                return "cold"

        # 최종 결과: member_id, user_segment만 반환
        result = user_stats.with_columns(
            pl.struct(user_stats.columns).map_elements(classify, return_dtype=pl.Utf8).alias("user_segment")
        ).select(["member_id", "user_segment"])

        return result

    def run(self, days: int = 7) -> pl.DataFrame:
        """원스톱: 최근 N일 로그 fetch → 세그먼트 분류 결과 반환"""
        all_users = self.conn.execute(_query_all_active_users())

        df = self.fetch_events_last_ndays(days=days)
        seg = self.classify_users_quantile(df)

        result = (
            all_users.join(seg, on="member_id", how="left")
                .with_columns(
                    pl.col("user_segment").fill_null("cold")
                )
        )
        return result
