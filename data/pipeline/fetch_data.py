import os
import random
import polars as pl
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict, Counter
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timezone, timedelta
from utils.logger import get_logger


# -----------------------------
# Popularity 기반 후보군 생성
# -----------------------------
def generate_popularity_candidates(
    days: int = 14,
    top_k: int = 50,
    weights: Dict[str, float] = {
        "like": 3.0,
        "share": 2.0,
        "bookmark": 1.5,
    }
) -> List[str]:
    """인기 있는 상위 K개의 아티클 ID를 생성합니다."""
    print()
    logger = get_logger(f"Generate Popular Top-{top_k}")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)

    logger.info("Caculating Popularity...")

    df = (
        pl.scan_parquet("data/raw/articles.parquet")
        .filter(pl.col("created_at") >= since)
        .select([
            "article_id",
            (pl.col("like_count") * weights["like"]).alias("like_score"),
            (pl.col("share_count") * weights["share"]).alias("share_score"),
            (pl.col("bookmark_count") * weights["bookmark"]).alias("bookmark_score")
        ])
        .with_columns([
            (pl.col("like_score") + pl.col("share_score") + pl.col("bookmark_score")).alias("popularity_score")
        ])
        .sort("popularity_score", descending=True)
        .select("article_id")
        .limit(top_k)
        .collect()
    )

    return df["article_id"].to_list()

# -----------------------------
# category_id → article_id mapping
# -----------------------------
def fetch_category_id_articles() -> Dict[int, Set[str]]:
    """카테고리 ID별로 매핑된 아티클 ID를 반환합니다."""
    df = pl.read_parquet("data/raw/articles.parquet")

    category_map = (
        df.group_by("category_id")
        .agg(pl.col("article_id"))
        .to_dict(as_series=False) 
    )

    return {
        cat_id: set(article_ids)
        for cat_id, article_ids in zip(category_map["category_id"], category_map["article_id"])
    }



# -----------------------------
# 전체 article_id 목록 추출
# -----------------------------
def fetch_all_article_ids() -> Set[str]:
    """RDB에서 전체 article_id 집합을 조회합니다."""
    df = pl.read_parquet("data/raw/articles.parquet")
    return set(df["article_id"])


# -----------------------------
# Positive + Negative 학습 샘플 생성
# -----------------------------
def generate_training_samples(
    user_positive_map: Dict[int, Set[str]],
    negative_ratio: int = 5,
    seed: int = 42
) -> List[Tuple[int, str, int]]:
    """Positive 로그와 전체 아티클 pool을 기반으로 negative 샘플을 생성합니다."""
    print()
    logger = get_logger("Generate Training Datasets")
    random.seed(seed)

    all_article_ids = fetch_all_article_ids()
    dataset = []

    logger.info("Creating Dataset using negative samples...")
    for user_id, pos_articles in user_positive_map.items():
        dataset.extend([(user_id, aid, 1) for aid in pos_articles])

        neg_pool = list(all_article_ids - pos_articles)
        sampled_negatives = random.sample(neg_pool, min(len(pos_articles) * negative_ratio, len(neg_pool)))
        dataset.extend([(user_id, aid, 0) for aid in sampled_negatives])

    return dataset

# -----------------------------
# 최근 3일 간 조회했던 게시글 조회
# -----------------------------
def fetch_recently_viewed_articles(conn: Any, days: int = 3) -> Dict[int, Set[str]]:
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    return fetch_positive_logs(conn, start_ts, end_ts, positive_events={"article_in"})


# -----------------------------
# Positive 로그 추출 from DynamoDB
# -----------------------------
def fetch_positive_logs(
    conn: Any,
    start_ts: int,
    end_ts: int,
    positive_events = {"article_in", "like", "archive", "share"}
) -> Dict[int, Set[str]]:
    """DynamoDB에서 positive event 로그를 조회하여 유저별로 본 아티클 ID 집합을 반환합니다."""
    dynamo = conn.get_dynamo()
    table = dynamo.Table("event")

    query = conn.execute("SELECT DISTINCT member_id FROM member")
    member_ids = list(query["member_id"])

    TARGET_TYPE = "article"

    user_article_map = defaultdict(set)
    for member_id in member_ids:
        last_evaluated_key = None
        while True:
            query_kwargs = {
                "KeyConditionExpression": Key("member_id").eq(member_id) & Key("timestamp").between(start_ts, end_ts),
                "FilterExpression": Attr("target_type").eq(TARGET_TYPE) & Attr("event_type").is_in(positive_events),
                "ProjectionExpression": "target_id"
            }
            if last_evaluated_key:
                query_kwargs["ExclusiveStartKey"] = last_evaluated_key

            response = table.query(**query_kwargs)
            for item in response.get("Items", []):
                article_id = convert_decimal_fields(item).get("target_id")
                if article_id:
                    user_article_map[member_id].add(article_id)

            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

    return user_article_map


# -----------------------------
# Decimal 변환 유틸
# -----------------------------
def convert_decimal_fields(item: dict) -> dict:
    """DynamoDB에서 반환된 Decimal 필드를 Python의 int/float로 변환합니다."""
    from decimal import Decimal
    return {
        k: int(v) if isinstance(v, Decimal) and v % 1 == 0 else float(v) if isinstance(v, Decimal) else v
        for k, v in item.items()
    }

def fetch_articles_to_parquet(conn: Any, output_path: str = "data/raw/articles.parquet") -> None:
    """
    RDS에서 Content Based Embedding을 위해 전체 article 데이터를 parquet으로 저장합니다.

    Args:
        conn (Any): pymysql 커넥션
        output_path (str): 저장할 parquet 파일 경로
    """
    print()
    logger = get_logger("Fetch Articles")

    query = """
        SELECT  
            article_id,
            blog_id,
            title,
            description,
            category_id,
            keywords,
            content_length,
            like_count,
            share_count,
            bookmark_count,
            lang,
            created_at
        FROM article
        WHERE category_id IS NOT NULL
        AND keywords IS NOT NULL
    """

    logger.info("Fetching article data from DB...")
    df = conn.execute(query)

    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df.fetchall(), schema=[col[0] for col in df.description])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Saving {df.shape[0]} articles to {output_path}")
    df.write_parquet(output_path, compression="zstd")

def build_preferred_category_map(
    conn: Any,
    user_positive_map: Dict[int, Set[str]],
    article_category_path: str = "data/raw/articles.parquet",
    top_n: int = 5
) -> Dict[int, Set[int]]:
    """
    유저별로 선호 카테고리 ID를 구성 (회원가입 시 선택 + 최근 행동 기반).

    Args:
        conn (Any): DB connection (for member_interest)
        user_positive_map (Dict[int, Set[str]]): 유저별 본 article_id 집합
        article_category_path (str): article_id → category_id 포함 parquet 경로
        top_n (int): 행동 기반 상위 카테고리 수

    Returns:
        Dict[int, Set[int]]: 유저별 선호 category_id 집합
    """

    # 1. article_id → category_id mapping
    df = pl.read_parquet(article_category_path).select(["article_id", "category_id"])
    article_to_cat = dict(zip(df["article_id"], df["category_id"]))

    # 2. DB에서 member_interest 테이블 조회
    query = "SELECT member_id, interest_id FROM member_interest"
    result = conn.execute(query)
    member_interest_map: Dict[int, Set[int]] = defaultdict(set)
    for row in result.iter_rows(named=True):
        member_id, interest_id = row['member_id'], row['interest_id']
        if member_id and interest_id:
            member_interest_map[member_id].add(int(interest_id))

    # 3. 각 유저별 preferred category 구성
    preferred_map = {}

    for user_id, article_ids in user_positive_map.items():
        cat_counter = Counter()
        for aid in article_ids:
            cat = article_to_cat.get(aid)
            if cat is not None:
                cat_counter[cat] += 1

        # 행동 기반: 최근 본 카테고리 상위 N개
        top_categories = [cat for cat, _ in cat_counter.most_common(top_n)]

        # 가입 기반: DB에서 선택한 관심사
        selected_interests = member_interest_map.get(user_id, set())

        # 합집합 → 전체 선호 category
        preferred = set(top_categories) | selected_interests
        preferred_map[user_id] = preferred

    return preferred_map
