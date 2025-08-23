from typing import Dict, List, Any
import polars as pl
from qdrant_client import QdrantClient
from data.utils import _query_all_active_users
from utils.logger import get_logger

logger = get_logger("MergeCandidates")

def get_active_users(conn):
    return conn.execute(_query_all_active_users())

def get_user_vector(qdrant: QdrantClient, member_id: int, user_collection: str = "user-profiles") -> List[float]:
    res = qdrant.retrieve(
        collection_name=user_collection,
        ids=[member_id],
        with_vectors=True
    )
    if not res:
        return []
    return res[0].vector

def search_personalized_items(qdrant: QdrantClient, member_id: int,
                              item_collection: str = "bite-vectordb",
                              user_collection: str = "user-profiles",
                              topk: int = 120) -> List[str]:
    uvec = get_user_vector(qdrant, member_id, user_collection)
    if not uvec:
        return []

    hits = qdrant.search(
        collection_name=item_collection,
        query_vector=uvec,
        limit=topk,
        with_payload=True
    )
    return [h.payload.get("article_id") for h in hits if h.payload and h.payload.get("article_id")]

def merge_candidates(
    personal: List[str],
    short_pop: List[str],
    long_pop: List[str],
    min_per_user: int = 100,
    cap_personal: int = 100,
    cap_short: int = 30,
    cap_long: int = 20,
) -> List[str]:
    """
    개인화/단기/장기 후보 합치고 중복 제거.
    우선순위: personal > short > long
    """
    seen, merged = set(), []

    for aid in personal[:cap_personal]:
        if aid not in seen:
            seen.add(aid)
            merged.append(aid)

    for aid in short_pop[:cap_short]:
        if aid not in seen:
            seen.add(aid)
            merged.append(aid)

    for aid in long_pop[:cap_long]:
        if aid not in seen:
            seen.add(aid)
            merged.append(aid)

    # 부족하면 personal 여분으로 채우기
    if len(merged) < min_per_user:
        for aid in personal[cap_personal:]:
            if len(merged) >= min_per_user:
                break
            if aid not in seen:
                seen.add(aid)
                merged.append(aid)

    return merged

def merge_candidates_for_all_users(
    conn: Any,
    qdrant: QdrantClient,
    popular: List[pl.DataFrame],
    min_per_user: int = 100,
    cap_personal: int = 100,
    cap_short: int = 30,
    cap_long: int = 20,
) -> pl.DataFrame:
    """
    전체 유저에 대해 후보 생성.
    반환: DataFrame {member_id, article_id, source, num_recs}
    """
    # 인기 리스트 준비
    short_list = popular[0].select(pl.col("article_id").cast(pl.Utf8)).to_series().to_list()
    long_list  = popular[1].select(pl.col("article_id").cast(pl.Utf8)).to_series().to_list()

    # 활성 유저 조회
    user_ids = get_active_users(conn).select(pl.col("member_id")).to_series().to_list()

    all_rows = []
    for uid in user_ids:
        personal = search_personalized_items(
            qdrant=qdrant,
            member_id=uid,
            item_collection="bite-vectordb",
            topk=120
        )
        merged = merge_candidates(
            personal=personal,
            short_pop=short_list,
            long_pop=long_list,
            min_per_user=min_per_user,
            cap_personal=cap_personal,
            cap_short=cap_short,
            cap_long=cap_long,
        )

        # 행 단위로 쌓기
        for aid in merged:
            all_rows.append({
                "member_id": uid,
                "article_id": aid,
            })

    df = pl.from_dicts(all_rows).with_columns([
        pl.col("member_id").cast(pl.Int64),
        pl.col("article_id").cast(pl.Utf8),
    ])
    return df