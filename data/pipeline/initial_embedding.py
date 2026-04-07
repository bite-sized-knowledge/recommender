import uuid
from typing import Dict, List
import numpy as np
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from data.utils import (
    _l2_normalize, _to_point_id, _scroll_centroid,
    ITEM_COLLECTION, USER_COLLECTION, CAT_COLLECTION, UUID_NAMESPACE, QDRANT_BATCH,
)
from utils.logger import get_logger

logger = get_logger("InitialEmbedding")

W_CATEGORY = 0.6
W_BLOG = 0.4
MIN_BLOG_ARTICLES = 3

def fetch_user_interests(conn) -> Dict[int, List[int]]:
    """
    member_interest(member_id INT, interest_id INT)
    return: {member_id: [interest_id, ...]}
    """
    res = conn.execute("""
        SELECT mi.member_id, mi.interest_id
        FROM member_interest mi
        JOIN member m ON mi.member_id = m.member_id
        WHERE m.status = 'active'
        AND m.role IN ('ROLE_USER', 'ROLE_GUEST')
    """)
    out = defaultdict(list)

    for mid, cid in res.select(["member_id", "interest_id"]).iter_rows():
        out[mid].append(cid)
    return out

def fetch_user_blog_subscriptions(conn) -> Dict[int, List[int]]:
    """
    blog_subscribe(blog_id, member_id, is_deleted)
    return: {member_id: [blog_id, ...]}
    """
    res = conn.execute("""
        SELECT bs.member_id, bs.blog_id
        FROM blog_subscribe bs
        JOIN member m ON bs.member_id = m.member_id
        WHERE bs.is_deleted = 0
        AND m.status = 'ACTIVE'
        AND m.role IN ('ROLE_USER', 'ROLE_GUEST')
    """)
    out = defaultdict(list)
    if res.is_empty():
        return out
    for mid, bid in res.select(["member_id", "blog_id"]).iter_rows():
        out[mid].append(bid)
    return out

def load_category_vectors(client: QdrantClient) -> Dict[int, np.ndarray]:
    cat_ids = list(range(1, 14))
    vecs: Dict[int, np.ndarray] = {}

    points = client.retrieve(collection_name=CAT_COLLECTION, ids=cat_ids, with_vectors=True)
    for p in points:
        vecs[int(p.id)] = np.asarray(p.vector, dtype=np.float32)
    if len(vecs) == 0:
        raise RuntimeError("category-profiles에서 벡터를 찾지 못했습니다.")
    return vecs

def compute_blog_centroid(client: QdrantClient, blog_id: int) -> np.ndarray | None:
    """Compute centroid of all article vectors belonging to a blog."""
    filt = Filter(must=[FieldCondition(key="blog_id", match=MatchValue(value=blog_id))])
    return _scroll_centroid(client, ITEM_COLLECTION, filt, min_vecs=MIN_BLOG_ARTICLES)

def compute_popular_centroid(conn, client: QdrantClient) -> np.ndarray | None:
    """Compute centroid from highly-engaged articles as a global popularity fallback."""
    res = conn.execute("""
        SELECT CAST(article_id AS CHAR) AS article_id
        FROM user_article_engagement
        WHERE engagement_score > 3.0
        ORDER BY engagement_score DESC
        LIMIT 200
    """)
    if res.is_empty():
        # Fallback to like_count
        res = conn.execute("""
            SELECT article_id
            FROM article
            WHERE like_count + bookmark_count > 0
            ORDER BY (like_count + bookmark_count) DESC
            LIMIT 200
        """)
    if res.is_empty():
        return None

    article_ids = res["article_id"].to_list()
    point_ids = [str(_to_point_id(aid, UUID_NAMESPACE)) for aid in article_ids]

    vecs = []
    for i in range(0, len(point_ids), QDRANT_BATCH):
        chunk = point_ids[i:i + QDRANT_BATCH]
        points = client.retrieve(
            collection_name=ITEM_COLLECTION,
            ids=chunk,
            with_vectors=True,
            with_payload=False,
        )
        for p in points:
            if p.vector is not None:
                vecs.append(np.asarray(p.vector, dtype=np.float32))

    if not vecs:
        return None
    return _l2_normalize(np.vstack(vecs).mean(axis=0).astype(np.float32))


def build_user_vector(cat_vecs: Dict[int, np.ndarray], picked: List[int]) -> np.ndarray | None:
    picked = [c for c in picked if c in cat_vecs]
    if not picked:
        return None
    mat = np.stack([cat_vecs[c] for c in picked], axis=0)
    return _l2_normalize(mat.mean(axis=0).astype(np.float32))

def build_user_initial_embedding(conn, client: QdrantClient, dim) -> Dict[int, np.ndarray]:

    # 카테고리 벡터 1회 로드
    cat_vecs = load_category_vectors(client)

    # user-profiles 컬렉션 없으면 생성
    if not client.collection_exists(USER_COLLECTION):
        client.create_collection(
            collection_name=USER_COLLECTION,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            ),
        )

    # 유저 관심 카테고리 로드
    user_to_cats = fetch_user_interests(conn)

    # 블로그 구독 로드
    user_to_blogs = fetch_user_blog_subscriptions(conn)

    # 블로그 centroid 캐시 (blog_id → vector)
    blog_centroid_cache: Dict[int, np.ndarray | None] = {}

    # 글로벌 popularity centroid (1회 계산)
    popular_centroid = compute_popular_centroid(conn, client)
    if popular_centroid is not None:
        logger.info("Global popular centroid computed for cold start fallback")

    # 유저별 벡터 계산
    all_user_ids = set(user_to_cats.keys()) | set(user_to_blogs.keys())

    # 관심사도 구독도 없는 유저 포함
    from data.utils import _query_all_active_users
    active_users = conn.execute(_query_all_active_users())
    for uid in active_users["member_id"].to_list():
        all_user_ids.add(uid)

    user_vecs: Dict[int, np.ndarray] = {}
    for mid in all_user_ids:
        cats = user_to_cats.get(mid, [])
        blogs = user_to_blogs.get(mid, [])

        cat_vec = build_user_vector(cat_vecs, cats)

        # Blog subscription centroid
        blog_vec = None
        if blogs:
            blog_vecs = []
            for bid in blogs:
                if bid not in blog_centroid_cache:
                    blog_centroid_cache[bid] = compute_blog_centroid(client, bid)
                bv = blog_centroid_cache[bid]
                if bv is not None:
                    blog_vecs.append(bv)
            if blog_vecs:
                blog_vec = _l2_normalize(np.vstack(blog_vecs).mean(axis=0).astype(np.float32))

        # Blend category + blog
        if cat_vec is not None and blog_vec is not None:
            vec = _l2_normalize((W_CATEGORY * cat_vec + W_BLOG * blog_vec).astype(np.float32))
        elif cat_vec is not None:
            vec = cat_vec
        elif blog_vec is not None:
            vec = blog_vec
        elif popular_centroid is not None:
            # Phase 1B: popularity fallback for truly cold users
            vec = popular_centroid
        else:
            continue

        user_vecs[mid] = vec

    return user_vecs
