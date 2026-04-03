import numpy as np
import polars as pl
from typing import Optional
from data.utils import EVENT_WEIGHTS
import datetime as dt


now_utc = dt.datetime.now()

def to_epoch_ms(ts: dt.datetime) -> int:
    return int(ts.timestamp() * 1000)

def fetch_article_in_counts(conn, start, end) -> pl.DataFrame:
    """
    MySQL user_events에서 event_type='ARTICLE_IN'인 이벤트를
    [start일 전, end일 전) 구간으로 조회하여 article_id별 count를 집계.
    반환 스키마: {article_id: Utf8, article_ins: Int64}
    """

    sql = f"""
    SELECT
        CAST(article_id AS CHAR) AS article_id,
        COUNT(*) AS article_ins
    FROM user_events
    WHERE LOWER(event_type) = 'article_in'
      AND occurred_at BETWEEN NOW() - INTERVAL {start} DAY AND NOW() - INTERVAL {end} DAY
      AND article_id IS NOT NULL
    GROUP BY article_id
    """

    df = conn.execute(sql)

    if df.is_empty():
        return pl.DataFrame(schema={"article_id": pl.Utf8, "article_ins": pl.Int64})

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

def compute_popular(conn, start, end) -> pl.DataFrame:
    df_in = fetch_article_in_counts(conn, start, end)                 # {article_id, article_ins}
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
