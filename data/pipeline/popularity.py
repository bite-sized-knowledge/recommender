import numpy as np
import polars as pl
from typing import Optional
from data.utils import (
    TOPK, EVENT_WEIGHTS, 
)
import datetime as dt
from boto3.dynamodb.conditions import Attr, Key


now_utc = dt.datetime.now()

def to_epoch_ms(ts: dt.datetime) -> int:
    return int(ts.timestamp() * 1000)

def fetch_article_in_counts(table, start, end) -> pl.DataFrame:
    """
    DynamoDB에서 event_type='article_in', target_type='article'인 이벤트를
    [START_MS, END_MS) 구간으로 스캔하여 article_id별 count를 집계.
    반환 스키마: {article_id: Utf8, article_ins: Int64}
    """

    START_MS = to_epoch_ms(now_utc - dt.timedelta(days=start))
    END_MS = to_epoch_ms(now_utc - dt.timedelta(days=end))

    resp = table.query(
        IndexName="event_type-timestamp-index",
        KeyConditionExpression=(
            Key("event_type").eq("article_in") &
            Key("timestamp").between(START_MS, END_MS)
        ),
        FilterExpression=Attr("target_type").eq("article")
    )

    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            IndexName="event_type-timestamp-index",
            KeyConditionExpression=(
                Key("event_type").eq("article_in") &
                Key("timestamp").between(START_MS, END_MS)
            ),
            FilterExpression=Attr("target_type").eq("article"),
            ExclusiveStartKey=resp["LastEvaluatedKey"]
        )
        items.extend(resp.get("Items", []))

    if not items:
        return pl.DataFrame(schema={"article_id": pl.Utf8, "article_ins": pl.Int64})

    df = pl.DataFrame(items)

    if "target_id" in df.columns:
        df = df.rename({"target_id": "article_id"})

    df = df.group_by("article_id").len().rename({"len": "article_ins"})
    return df

def fetch_engagements(conn, start, end) -> pl.DataFrame:
    """
    MySQL에서 3–7일 구간 like/share/bookmark 집계.
    반환 스키마: {article_id: Utf8, likes: Int64, shares: Int64, bookmarks: Int64}
    """

    end_dt = now_utc - dt.timedelta(days=end)
    start_dt = now_utc - dt.timedelta(days=start)


    SQL = f"""
    WITH ev AS (
        SELECT article_id, COUNT(*) AS like_cnt, 0 AS share_cnt, 0 AS bookmark_cnt
        FROM article_like
        WHERE is_deleted = 0
          AND created_at >= '{start_dt}' AND created_at < '{end_dt}'
        GROUP BY article_id
        UNION ALL
        SELECT article_id, 0, COUNT(*), 0
        FROM article_share
        WHERE 
          created_at >= '{start_dt}' AND created_at < '{end_dt}'
        GROUP BY article_id
        UNION ALL
        SELECT article_id, 0, 0, COUNT(*)
        FROM article_bookmark
        WHERE is_deleted = 0
          AND created_at >= '{start_dt}' AND created_at < '{end_dt}'
        GROUP BY article_id
    )
    SELECT
      CAST(article_id AS CHAR) AS article_id,
      SUM(like_cnt)     AS likes,
      SUM(share_cnt)    AS shares,
      SUM(bookmark_cnt) AS bookmarks
    FROM ev
    GROUP BY article_id
    """

    df = conn.execute(SQL)

    if df.is_empty():
        return pl.DataFrame(schema={
            "article_id": pl.Utf8, "likes": pl.Int64, "shares": pl.Int64, "bookmarks": pl.Int64
        })

    return df

def compute_popular(conn ,table, start, end) -> pl.DataFrame:
    df_in = fetch_article_in_counts(table, start, end)                 # {article_id, article_ins}
    df_eng = fetch_engagements(conn, start, end)                      # {article_id, likes, shares, bookmarks}

    # full-outer join: 특정 소스에만 존재하는 article도 살림
    df = df_eng.join(df_in, on="article_id", how="outer")
    df = (
        df.with_columns([
            pl.coalesce([pl.col("article_id"), pl.col("article_id_right")]).alias("article_id_merged")
        ])
        .drop(["article_id", "article_id_right"])
        .rename({"article_id_merged": "article_id"})
    )

    # null → 0
    df = df.with_columns([
        pl.col("likes").fill_null(0),
        pl.col("shares").fill_null(0),
        pl.col("bookmarks").fill_null(0),
        pl.col("article_ins").fill_null(0),
    ])

    # 점수: log1p로 카운트 스케일 눌러줌, article_in은 sqrt로 과대반영 방지
    df = df.with_columns([
        pl.when(pl.col("article_ins") > 0)
          .then(pl.col("article_ins").cast(pl.Float64).sqrt())
          .otherwise(0.0)
          .alias("in_term"),
        pl.col("likes").cast(pl.Float64).log1p().alias("likes_term"),
        pl.col("shares").cast(pl.Float64).log1p().alias("shares_term"),
        pl.col("bookmarks").cast(pl.Float64).log1p().alias("bookmarks_term"),
    ])

    df = df.with_columns([
        (
            EVENT_WEIGHTS["like"]      * pl.col("likes_term") +
            EVENT_WEIGHTS["share"]     * pl.col("shares_term") +
            EVENT_WEIGHTS["archive"]  * pl.col("bookmarks_term") +
            EVENT_WEIGHTS["article_in"]* pl.col("in_term")
        ).alias("score")
    ])

    # 내림차순 정렬, TOPK
    df = df.sort("score", descending=True).limit(80)

    return df.select(
        "article_id"
    )