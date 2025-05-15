import polars as pl
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from boto3.dynamodb.conditions import Key


def fetch_articles_to_parquet(conn):
    """
    Fetch articles from the RDS database and save them as a compressed Parquet file.
    This function filters out articles that have null values in `category_id` or `keywords`.

    Args:
        conn: A database connection object.

    Returns:
        None. The result is saved to 'data/raw/articles.parquet'.
    """
    df = conn.execute("""
        SELECT  
            article_id,
            blog_id,
            title,
            description,
            category_id,
            keywords,
            content_length,
            lang
        FROM article
        WHERE category_id IS NOT NULL
          AND keywords IS NOT NULL
    """)

    print("Saving Articles to data/raw/articles.parquet ...")
    df.write_parquet("data/raw/articles.parquet", compression="zstd")


def convert_decimal_fields(item):
    """
    Convert all Decimal fields in a dictionary to integers (used for DynamoDB data).

    Args:
        item (dict): A dictionary representing a DynamoDB item.

    Returns:
        dict: The same dictionary with Decimal fields converted to integers.
    """
    return {
        k: int(v) if isinstance(v, Decimal) else v for k, v in item.items()
    }


def fetch_events_to_parquet(conn):
    """
    Fetch 'like', 'blog_in', and 'archive' events from the last 3 days from DynamoDB,
    deduplicate them, and save to a compressed Parquet file.

    Deduplication is based on (member_id, target_id, event_type) triplets.

    Args:
        conn: An object with a `get_dynamo()` method that returns a boto3 DynamoDB resource.

    Returns:
        None. The result is saved to 'data/raw/events.parquet'.
    """
    dynamo = conn.get_dynamo()
    table = dynamo.Table("event")

    three_days_ago = int((datetime.now(timezone.utc) - timedelta(days=3)).timestamp())
    all_events = []
    seen = set()

    for event_type in ["like", "blog_in", "archive"]:
        last_evaluated_key = None
        while True:
            query_params = {
                "IndexName": "event_type-timestamp-index",
                "KeyConditionExpression": Key("event_type").eq(event_type) & Key("timestamp").gte(three_days_ago),
            }
            if last_evaluated_key:
                query_params["ExclusiveStartKey"] = last_evaluated_key

            response = table.query(**query_params)
            items = response.get("Items", [])

            for item in items:
                item = convert_decimal_fields(item)
                key = (item.get("member_id"), item.get("target_id"), item.get("event_type"))
                if key not in seen:
                    seen.add(key)
                    all_events.append({
                        "member_id": item.get("member_id"),
                        "article_id": item.get("target_id"),
                        "event_type": item.get("event_type")
                    })

            last_evaluated_key = response.get("LastEvaluatedKey", None)
            if not last_evaluated_key:
                break

    df = pl.DataFrame(all_events)
    print("Saving Events to data/raw/events.parquet ...")
    df.write_parquet("data/raw/events.parquet", compression="zstd")
