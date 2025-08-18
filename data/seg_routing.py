import time
from typing import Any, Dict, List, Optional
from boto3.dynamodb.conditions import Attr
import polars as pl
from data.utils import (
    _to_py, _ensure_keys, _query_all_active_users
)

class UserSegmentation:
    """
    DynamoDB 로그 기반 유저 세그먼트 분류 (cold / warm / hot)
    - 최근 N일 로그 스캔 → Polars df
    - 분류 기준: Recency(최근 1/3/7일), 이벤트 수 Quantile, 행동 다양성
    """

    def __init__(
        self,
        conn = None
    ):
        if conn is None:
            raise ValueError("table_resource를 반드시 제공해야 합니다.")

        dynamo = conn.get_dynamo() 
        self.tbl = dynamo.Table('event')
        self.conn = conn

    # -------------------- Public API --------------------

    def fetch_events_last_ndays(self, days: int = 7, limit: int = 1000) -> pl.DataFrame:
        """최근 N일 로그를 스캔하여 Polars DataFrame 반환"""
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000

        expr_attr_names = {"#ts": "timestamp"}
        projection_expression = "member_id, event_type, target_type, target_id, #ts"
        filter_expression = Attr("timestamp").between(start_ms, end_ms)

        items: List[Dict[str, Any]] = []
        last_evaluated_key: Optional[Dict[str, Any]] = None

        while True:
            scan_kwargs = {
                "FilterExpression": filter_expression,
                "ExpressionAttributeNames": expr_attr_names,
                "ProjectionExpression": projection_expression,
                "Limit": limit,
            }
            if last_evaluated_key:
                scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

            resp = self.tbl.scan(**scan_kwargs)
            page_items = resp.get("Items", [])
            if page_items:
                items.extend(page_items)

            last_evaluated_key = resp.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

        if not items:
            return pl.DataFrame(
                schema={
                    "member_id": pl.Int64,
                    "event_type": pl.Utf8,
                    "target_type": pl.Utf8,
                    "target_id": pl.Utf8,
                    "timestamp": pl.Int64,
                    "datetime_utc": pl.Datetime("ms"),
                }
            )

        wanted = ["member_id", "event_type", "target_type", "target_id", "timestamp"]
        items_norm = [_ensure_keys(_to_py(it), wanted) for it in items]

        df = pl.from_dicts(
            items_norm,
            schema={
                "member_id": pl.Int64,
                "event_type": pl.Utf8,
                "target_type": pl.Utf8,
                "target_id": pl.Utf8,
                "timestamp": pl.Int64,  # epoch ms
            },
        ).with_columns(
            pl.col("member_id").cast(pl.Int64, strict=False),
            pl.col("event_type").cast(pl.Utf8, strict=False),
            pl.col("target_type").cast(pl.Utf8, strict=False),
            pl.col("target_id").cast(pl.Utf8, strict=False),
            pl.col("timestamp").cast(pl.Int64, strict=False),
        )

        # ms → Datetime[ms, UTC]
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), "ms")
              .dt.replace_time_zone("UTC")
              .alias("datetime_utc")
        )

        # 안전 필터 (범위 밖 제거)
        df = df.filter(
            pl.col("timestamp").is_not_null()
            & (pl.col("timestamp") >= start_ms)
            & (pl.col("timestamp") <= end_ms)
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
