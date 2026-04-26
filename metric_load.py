import json
from datetime import datetime

from metrics.recall_calculator import RecallEvaluator
from common.db import Connection
from utils.metric_sink import upsert_recall_daily


if __name__ == "__main__":
    conn = None
    try:
        conn = Connection()

        evaluator = RecallEvaluator(
            conn=conn,
        )

        k_list = [10, 30, 50]
        result_json = evaluator.evaluate(
            k_list=k_list,
            use_db=True  # recommendation 테이블에서 직접 추출할 경우
        )

        print(result_json)

        # metric.bite-sized.xyz Recommendation 탭에서 시계열로 사용.
        try:
            metrics = json.loads(result_json)
            metric_date = datetime.strptime(metrics["metric_date"], "%Y-%m-%d").date()
            for k in k_list:
                recall = metrics.get(f"recall_at_{k}")
                if recall is None:
                    continue
                upsert_recall_daily(
                    conn=conn,
                    metric_date=metric_date,
                    k=k,
                    recall=recall,
                    hit_users=metrics.get(f"hit_users_at_{k}"),
                    total_users=metrics.get("total_users"),
                    total_recommendations=metrics.get("total_recommendations"),
                    unique_items=metrics.get("unique_items_recommended"),
                )
        except Exception as persist_err:
            print(f"recall_daily upsert skipped: {persist_err}")

    except Exception as e:
         print(e)
         print("pipeline failed")

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
