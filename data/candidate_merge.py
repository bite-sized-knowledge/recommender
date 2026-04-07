from typing import Dict, List, Any, Tuple
import polars as pl
from qdrant_client import QdrantClient
from data.utils import _query_all_active_users, ITEM_COLLECTION, USER_COLLECTION
from utils.logger import get_logger

logger = get_logger("MergeCandidates")

SRC_PERSONALIZED = "personalized"
SRC_SHORT_POP = "short_pop"
SRC_LONG_POP = "long_pop"

def get_active_users(conn):
    return conn.execute(_query_all_active_users())

def get_user_vector(qdrant: QdrantClient, member_id: int, user_collection: str = USER_COLLECTION) -> List[float]:
    res = qdrant.retrieve(
        collection_name=user_collection,
        ids=[member_id],
        with_vectors=True
    )
    if not res:
        return []
    return res[0].vector

def search_personalized_items(qdrant: QdrantClient, member_id: int,
                              item_collection: str = ITEM_COLLECTION,
                              user_collection: str = USER_COLLECTION,
                              topk: int = 120) -> List[Tuple[str, float]]:
    """Returns list of (article_id, cosine_similarity_score)."""
    uvec = get_user_vector(qdrant, member_id, user_collection)
    if not uvec:
        return []

    hits = qdrant.search(
        collection_name=item_collection,
        query_vector=uvec,
        limit=topk,
        with_payload=True
    )
    return [
        (h.payload.get("article_id"), h.score)
        for h in hits
        if h.payload and h.payload.get("article_id")
    ]

def merge_candidates(
    personal: List[Tuple[str, float]],
    short_pop: List[Tuple[str, float]],
    long_pop: List[Tuple[str, float]],
    min_per_user: int = 100,
    cap_personal: int = 100,
    cap_short: int = 30,
    cap_long: int = 20,
) -> List[Tuple[str, float, str]]:
    """
    Merge candidates with dedup. Returns [(article_id, score, source), ...].
    Priority: personal > short > long
    """
    seen, merged = set(), []

    sources = [
        (personal[:cap_personal], SRC_PERSONALIZED),
        (short_pop[:cap_short], SRC_SHORT_POP),
        (long_pop[:cap_long], SRC_LONG_POP),
    ]
    for items, label in sources:
        for aid, score in items:
            if aid not in seen:
                seen.add(aid)
                merged.append((aid, score, label))

    if len(merged) < min_per_user:
        for aid, score in personal[cap_personal:]:
            if len(merged) >= min_per_user:
                break
            if aid not in seen:
                seen.add(aid)
                merged.append((aid, score, SRC_PERSONALIZED))

    return merged

def merge_candidates_for_all_users(
    conn: Any,
    qdrant: QdrantClient,
    popular: List[pl.DataFrame],
    segments: pl.DataFrame = None,
    min_per_user: int = 100,
) -> pl.DataFrame:
    """
    Generate candidates for all users with segment-aware caps.
    Returns DataFrame {member_id, article_id, score, source}
    """
    # Popularity lists with scores
    short_df = popular[0]
    long_df = popular[1]

    short_list = [
        (row["article_id"], row["score"] if "score" in short_df.columns else 0.0)
        for row in short_df.iter_rows(named=True)
    ]
    long_list = [
        (row["article_id"], row["score"] if "score" in long_df.columns else 0.0)
        for row in long_df.iter_rows(named=True)
    ]

    # Build segment lookup
    seg_map: Dict[int, str] = {}
    if segments is not None and not segments.is_empty():
        for row in segments.iter_rows(named=True):
            seg_map[row["member_id"]] = row["user_segment"]

    # Segment-dependent caps
    SEGMENT_CAPS = {
        "cold": {"cap_personal": 40, "cap_short": 40, "cap_long": 20},
        "warm": {"cap_personal": 80, "cap_short": 25, "cap_long": 15},
        "hot":  {"cap_personal": 100, "cap_short": 15, "cap_long": 5},
    }
    DEFAULT_CAPS = SEGMENT_CAPS["warm"]

    user_ids = get_active_users(conn).select(pl.col("member_id")).to_series().to_list()

    all_rows = []
    for uid in user_ids:
        personal = search_personalized_items(
            qdrant=qdrant,
            member_id=uid,
            item_collection=ITEM_COLLECTION,
            topk=120
        )

        seg = seg_map.get(uid, "cold")
        caps = SEGMENT_CAPS.get(seg, DEFAULT_CAPS)

        merged = merge_candidates(
            personal=personal,
            short_pop=short_list,
            long_pop=long_list,
            min_per_user=min_per_user,
            **caps,
        )

        for aid, score, source in merged:
            all_rows.append({
                "member_id": uid,
                "article_id": aid,
                "score": score,
                "source": source,
            })

    df = pl.from_dicts(all_rows, schema={
        "member_id": pl.Int64,
        "article_id": pl.Utf8,
        "score": pl.Float64,
        "source": pl.Utf8,
    })
    return df
