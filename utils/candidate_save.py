import polars as pl
from datetime import datetime, timedelta
from data.utils import RETENTION_DAYS
from utils.logger import get_logger

ALLOWED_TABLES = {"recommendation"}

def save_recommendations_to_db(df, conn, table_name: str = "recommendation"):
    """
    Insert (member_id, article_id, score) pairs into the recommendation table.
    """
    logger = get_logger(f"Saving Data into {table_name} table...")

    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}")

    if df.is_empty():
        print("No recommendations to insert.")
        return

    threshold_date = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    logger.info(f"Deleting old records before {threshold_date}")
    conn._raw_execute(
        f"DELETE FROM {table_name} WHERE created_at < :threshold",
        {"threshold": threshold_date}
    )

    # Batch insert with parameterized queries
    insert_sql = f"INSERT INTO {table_name} (member_id, article_id, score) VALUES (:member_id, :article_id, :score)"

    has_score = "score" in df.columns
    param_list = []
    for row in df.iter_rows(named=True):
        param_list.append({
            "member_id": row["member_id"],
            "article_id": row["article_id"],
            "score": row.get("score") if has_score else None,
        })

    BATCH_SIZE = 500
    logger.info(f"Insert Query Start ({len(param_list)} rows)")
    for i in range(0, len(param_list), BATCH_SIZE):
        batch = param_list[i:i + BATCH_SIZE]
        conn._batch_execute(insert_sql, batch)
