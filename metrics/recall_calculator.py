import polars as pl
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import json


class RecallEvaluator:
    """
    추천 로그와 유저 클릭 이벤트를 활용하여 recall@k를 평가
    """
    def __init__(self, conn: Any, parquet_path: str = "metrics/recommendation_logs.parquet"):
        self.conn = conn
        self.parquet_path = parquet_path

    def load_recommendations_from_db(self, target_date: str) -> None:
        """
        지정한 날짜의 추천 결과를 DB에서 불러와 parquet 파일로 저장
        """
        query = f"""
            SELECT member_id, article_id, recommendation_id, created_at
            FROM recommendation
            WHERE DATE_FORMAT(created_at, '%Y-%m-%d') = '{target_date}'
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
        지정한 유저와 기간에 대해 MySQL에서 클릭 이벤트를 조회
        """
        if not member_ids:
            return pl.DataFrame(schema={"member_id": pl.Int64, "target_id": pl.Utf8})

        ids_str = ",".join(str(mid) for mid in member_ids)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        sql = f"""
        SELECT
            member_id,
            CAST(article_id AS CHAR) AS target_id
        FROM user_events
        WHERE member_id IN ({ids_str})
          AND LOWER(event_type) = 'article_in'
          AND article_id IS NOT NULL
          AND occurred_at BETWEEN '{start_str}' AND '{end_str}'
        """

        df = self.conn.execute(sql)

        if df.is_empty():
            return pl.DataFrame(schema={"member_id": pl.Int64, "target_id": pl.Utf8})

        return df.select(["member_id", "target_id"])

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
        recommend_date: str = datetime.now().strftime("%Y-%m-%d"),
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
        rec_time = datetime.strptime(recommend_date, "%Y-%m-%d")
        end_time = rec_time + timedelta(hours=24)

        print(f"Timestamp {rec_time}")

        click_df = self.fetch_click_events(member_ids, rec_time, end_time)
        metrics = self.calculate_recall_at_k(rec_df, click_df, k_list)
        metrics["metric_date"] = recommend_date
        metrics["created_at"] = datetime.now().isoformat()

        return json.dumps(metrics, ensure_ascii=False, indent=2)
