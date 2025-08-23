import polars as pl
from datetime import datetime, timedelta
from utils.logger import get_logger

def save_recommendations_to_db(df, conn, table_name: str = "recommendation"):
    """
    Insert (member_id, article_id) pairs from a merged parquet file into a recommendation table.

    Args:
        parquet_path (str): Path to the merged.parquet file
        connection (Connection): Database connection object with _raw_execute
        table_name (str): Table name to insert into
    """
    logger = get_logger(f"Saving Data into {table_name} table...")
    
    if df.is_empty():
        print("No recommendations to insert.")
        return
    
    threshold_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    logger.info(f"Deleting old records before {threshold_date}")
    delete_sql = f"""
        DELETE FROM {table_name}
        WHERE created_at < '{threshold_date}';
    """
    conn._raw_execute(delete_sql)


    values = df.iter_rows()
    values_str = ", ".join(
        f"({member_id}, '{article_id}')"
        for member_id, article_id in values
    )

    sql = f"""
        INSERT {table_name} (member_id, article_id)
        VALUES {values_str};
    """

    logger.info("Insert Query Start")
    conn._raw_execute(sql)
