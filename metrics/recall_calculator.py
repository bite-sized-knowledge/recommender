import polars as pl
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from boto3.dynamodb.conditions import Key, Attr
import json


class RecallEvaluator:
    """
    추천 로그와 유저 클릭 이벤트를 활용하여 recall@k를 평가
    """
    def __init__(self, conn: Any, dynamo: Any, parquet_path: str = "metrics/recommendation_logs.parquet"):
        self.dynamo = dynamo
        self.conn = conn
        self.parquet_path = parquet_path

    def load_recommendations_from_db(self, target_date: str) -> None:
        """
        지정한 날짜의 추천 결과를 DB에서 불러와 parquet 파일로 저장
        """
        query = f"""
            SELECT member_id, article_id, recommendation_id, created_at
            FROM recommendation
            WHERE DATE(created_at) = '{target_date}'
        """

        self.conn.save_parquet(query, self.parquet_path)

    def load_recommendations_from_parquet(self, target_date: str) -> pl.DataFrame:
        """
        지정한 날짜의 추천 결과를 parquet 파일 load
        """
        df = pl.read_parquet(self.parquet_path)
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        return df.filter(pl.col("created_at").cast(pl.Date).eq(target))

    def fetch_click_events(
        self,
        member_ids: List[int],
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """
        지정한 유저와 기간에 대해 DynamoDB에서 클릭 이벤트를 조회
        """
        table = self.dynamo.Table("event")
        items = []
        # DynamoDB는 range key로 batch query를 지원하지 않으므로 유저별로 조회
        for member_id in member_ids:
            try:
                response = table.query(
                    KeyConditionExpression=Key("member_id").eq(member_id) &
                                           Key("timestamp").between(int(start_time.timestamp()), int(end_time.timestamp())),
                    FilterExpression=Attr("event_type").eq("article_in") & Attr("target_type").eq("article")
                )
                items.extend(response.get("Items", []))
            except Exception as e:
                # 유저별 조회 중 오류 발생 시 로그 출력 후 계속 진행
                print(f"DynamoDB 조회 오류 (member_id: {member_id}): {e}")

        if not items:
            return pl.DataFrame(schema={"member_id": pl.Int64, "target_id": pl.Utf8})

        # member_id, target_id 컬럼이 존재하는지 확인 후 반환
        df = pl.DataFrame(items)
        if "member_id" in df.columns and "target_id" in df.columns:
            return df.select(["member_id", "target_id"])
        else:
            # 컬럼이 없을 경우 빈 DataFrame 반환
            return pl.DataFrame(schema={"member_id": pl.Int64, "target_id": pl.Utf8})

    def calculate_recall_at_k(
        self,
        rec_df: pl.DataFrame,
        click_df: pl.DataFrame,
        k_list: List[int]
    ) -> Dict[str, Any]:
        """
        추천 결과와 클릭 이벤트를 바탕으로 recall@k를 계산
        """
        metrics = {
            "total_users": rec_df.select("member_id").unique().height,
            "total_recommendations": rec_df.height,
            "unique_items_recommended": rec_df.select("article_id").unique().height,
        }

        # 유저별로 추천 리스트 그룹화
        grouped = rec_df.sort(["member_id", "recommendation_id"]).group_by("member_id").agg([
            pl.col("article_id").alias("recommendation_list")
        ])
        rec_map = {row["member_id"]: row["recommendation_list"] for row in grouped.iter_rows(named=True)}

        # 유저별 클릭 아이템 집합 생성
        click_grouped = click_df.group_by("member_id").agg(pl.col("target_id"))
        click_map = {row["member_id"]: set(row["target_id"]) for row in click_grouped.iter_rows(named=True)}

        for k in k_list:
            hit_users = 0
            for user_id, recs in rec_map.items():
                top_k = recs[:k]
                clicked_items = click_map.get(user_id, set())
                if set(top_k) & clicked_items:
                    hit_users += 1
            metrics[f"recall_at_{k}"] = round(hit_users / metrics["total_users"], 4) if metrics["total_users"] else 0.0
            metrics[f"hit_users_at_{k}"] = hit_users

        return metrics

    def evaluate(
        self,
        recommend_date: str,
        k_list: List[int] = [10, 30, 50, 100],
        use_db: bool = False
    ) -> str:
        """
        지정한 날짜에 대해 recall 평가를 수행하고 JSON 문자열로 반환
        """
        if use_db:
            self.load_recommendations_from_db(recommend_date)

        rec_df = self.load_recommendations_from_parquet(recommend_date)
        if rec_df.is_empty():
            raise ValueError(f"{recommend_date} 날짜의 추천 데이터가 없습니다.")

        member_ids = rec_df.select("member_id").unique().to_series().to_list()
        rec_time = datetime.strptime(recommend_date, "%Y-%m-%d").replace(hour=2, tzinfo=timezone.utc)
        end_time = rec_time + timedelta(days=1)

        click_df = self.fetch_click_events(member_ids, rec_time, end_time)
        metrics = self.calculate_recall_at_k(rec_df, click_df, k_list)
        metrics["metric_date"] = recommend_date
        metrics["created_at"] = datetime.now().isoformat()

        return json.dumps(metrics, ensure_ascii=False, indent=2)
