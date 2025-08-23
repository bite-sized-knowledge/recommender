from typing import Dict, List, Iterable
import numpy as np
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from data.utils import _l2_normalize

# Config 
CAT_COLLECTION = "category-profiles"
USR_COLLECTION = "user-profiles"
BATCH_UPSERT = 2000

# 유틸
def batched(iterable: Iterable, n: int) -> Iterable[list]:
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

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

def load_category_vectors(client: QdrantClient) -> Dict[int, np.ndarray]:
    cat_ids = list(range(1, 14))
    vecs: Dict[int, np.ndarray] = {}

    points = client.retrieve(collection_name=CAT_COLLECTION, ids=cat_ids, with_vectors=True)
    for p in points:
        vecs[int(p.id)] = np.asarray(p.vector, dtype=np.float32)
    if len(vecs) == 0:
        raise RuntimeError("category-profiles에서 벡터를 찾지 못했습니다.")
    return vecs

def build_user_vector(cat_vecs: Dict[int, np.ndarray], picked: List[int]) -> np.ndarray | None:
    picked = [c for c in picked if c in cat_vecs]
    if not picked:
        return None
    mat = np.stack([cat_vecs[c] for c in picked], axis=0)
    return _l2_normalize(mat.mean(axis=0).astype(np.float32))

def build_user_initial_embedding(conn, client:QdrantClient, dim) -> Dict[int, List[float]]:

    # 카테고리 벡터 1회 로드 
    cat_vecs = load_category_vectors(client)

    # user-profiles 컬렉션 없으면 생성
    if not client.collection_exists(USR_COLLECTION):
        client.create_collection(
            collection_name=USR_COLLECTION,
            vectors_config=VectorParams(
                size=dim, 
                distance=Distance.COSINE
            ),
        )

    # 유저 관심 카테고리 로드
    user_to_cats = fetch_user_interests(conn)

    # 유저별 벡터 계산
    user_vecs: Dict[int, List[float]] = {}
    for mid, cats in user_to_cats.items():
        vec = build_user_vector(cat_vecs, cats)
        if vec is not None:
            user_vecs[mid] = vec

    return user_vecs