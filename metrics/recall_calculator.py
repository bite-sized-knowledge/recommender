import polars as pl
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from boto3.dynamodb.conditions import Key, Attr
from boto3.resources.base import ServiceResource

def load_recommendations_from_db(conn: Any, target_date: str) -> pl.DataFrame:
    """
    MySQL에서 recommendation 테이블을 불러와 Polars DataFrame으로 변환
    """
    query = f"""
        SELECT member_id, article_id, recommendation_id, created_at
        FROM recommendation
        WHERE DATE(created_at) = '{target_date}'
    """
    conn.save_parquet(query, "metrics/recommendation_logs.parquet")

def filter_recommendations_by_date(df: pl.DataFrame, target_date: str) -> pl.DataFrame:
    """
    특정 날짜(created_at 기준)의 추천 데이터만 필터링
    """
    return df.filter(pl.col("created_at").str.slice(0, 10) == target_date)


def fetch_click_events_from_dynamodb(
    dynamo: ServiceResource,
    member_ids: List[int],
    start_time: datetime,
    end_time: datetime,
) -> pl.DataFrame:
    """
    DynamoDB에서 클릭 로그(article_in)를 수집하여 Polars DataFrame으로 반환
    """
    table = dynamo.Table("events")
    items = []

    for member_id in member_ids:
        response = table.query(
            KeyConditionExpression=Key("member_id").eq(member_id) &
                                   Key("timestamp").between(int(start_time.timestamp()), int(end_time.timestamp())),
            FilterExpression=Attr("event_type").eq("article_in") & Attr("target_type").eq("article")
        )
        items.extend(response.get("Items", []))

    if not items:
        return pl.DataFrame(schema={"member_id": pl.Int64, "target_id": pl.Utf8})

    return pl.DataFrame(items).select(["member_id", "target_id"])


def calculate_recall_at_k(
    rec_df: pl.DataFrame,
    click_df: pl.DataFrame,
    k_list: List[int]
) -> Dict[str, Any]:
    """
    Recall@K 계산
    """
    metrics = {
        "total_users": rec_df.select("member_id").unique().height,
        "total_recommendations": rec_df.height,
        "unique_items_recommended": rec_df.select("article_id").unique().height,
    }

    # 유저별 추천 리스트 생성
    grouped = rec_df.sort(["member_id", "recommendation_id"]).group_by("member_id").agg([
        pl.col("article_id").alias("recommendation_list")
    ])

    rec_map = {row["member_id"]: row["recommendation_list"] for row in grouped.iter_rows()}
    click_map = click_df.group_by("member_id").agg(pl.col("target_id")).to_dict(as_series=False)

    for k in k_list:
        hit_users = 0
        for user_id, recs in rec_map.items():
            top_k = recs[:k]
            clicked_items = set(click_map.get("target_id", {}).get(user_id, []))
            if clicked_items & set(top_k):
                hit_users += 1
        metrics[f"recall_at_{k}"] = round(hit_users / metrics["total_users"], 4) if metrics["total_users"] else 0.0
        metrics[f"hit_users_at_{k}"] = hit_users

    return metrics


def calculate_daily_recall_metrics(
    conn: Any,
    parquet_path: str,
    recommend_date: str,
    dynamo: ServiceResource,
    k_list: List[int] = [10, 30, 50, 100]
) -> Dict[str, Any]:
    """
    전체 Recall 계산 흐름 제어
    """
    rec_df = load_recommendations(parquet_path)
    rec_df = filter_recommendations_by_date(rec_df, recommend_date)

    if rec_df.is_empty():
        raise ValueError(f"No recommendation data found for date {recommend_date}")

    member_ids = rec_df.select("member_id").unique().to_series().to_list()
    rec_time = datetime.strptime(recommend_date, "%Y-%m-%d").replace(hour=2, tzinfo=timezone.utc)
    end_time = rec_time + timedelta(days=1)

    click_df = fetch_click_events_from_dynamodb(dynamo, member_ids, rec_time, end_time)
    metrics = calculate_recall_at_k(rec_df, click_df, k_list)
    metrics["metric_date"] = recommend_date
    metrics["created_at"] = datetime.now().isoformat()

    return metrics
