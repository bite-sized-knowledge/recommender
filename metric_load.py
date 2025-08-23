from metrics.recall_calculator import RecallEvaluator 
from common.db import Connection


if __name__ == "__main__":
    try:
        conn = Connection()

        evaluator = RecallEvaluator(
            conn=conn,
            dynamo=conn.get_dynamo()
        )

        metrics = evaluator.evaluate(
            k_list=[10, 30, 50],
            use_db=True  # recommendation 테이블에서 직접 추출할 경우
        )

        print(metrics)

    except Exception as e:
         print(e)
         print("pipeline failed")

    finally:
        try:
            conn.close()
        except Exception:
            pass