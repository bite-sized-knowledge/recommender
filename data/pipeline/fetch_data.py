import numpy as np
from typing import Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from data.utils import _l2_normalize

ITEM_COLLECTION = "bite-vectordb"
CAT_COLLECTION  = "category-profiles"
CATEGORY = [i for i in range(1, 14)]
MIN_POINTS = 5


# ----- 유틸 -----
def get_dim_and_metric(client: QdrantClient, collection: str) -> Tuple[int, str]:
    info = client.get_collection(collection)
    dim = info.config.params.vectors.size
    dist = info.config.params.vectors.distance.value
    return dim, dist


# ----- 카테고리 수집/벡터 수집 -----
def fetch_vectors_by_category(client, category_value: int) -> np.ndarray:
    filt = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category_value))])
    vecs = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=ITEM_COLLECTION,
            limit=256, 
            with_payload=True, 
            with_vectors=True, 
            offset=next_offset,
            scroll_filter=filt
        )

        for p in points:
            v = p.vector
            if v is not None:
                vecs.append(np.asarray(v, dtype=np.float32))

        if next_offset is None:
            break

    return np.vstack(vecs) if vecs else np.zeros((0,))


def init_collection(client, collection: str, dim: int):
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )

# ----- 메인 로직 -----
def build_category_profiles(client, min_points: int = MIN_POINTS):
    dim, _ = get_dim_and_metric(client, ITEM_COLLECTION)      # 아이템 컬렉션 기준으로 맞춤l;
    init_collection(
        client,
        CAT_COLLECTION,
        dim
    )

    created, skipped = 0, 0

    for c in CATEGORY:
        V = fetch_vectors_by_category(client, c)
        if V.shape[0] < min_points:
            skipped += 1
            continue

        # 아이템 벡터가 이미 정규화라면 단순 평균 후 재정규화
        centroid = _l2_normalize(V.mean(axis=0))

        client.upsert(
            collection_name=CAT_COLLECTION,
            points=[
                PointStruct(
                    id=c,
                    vector=centroid.tolist(),
                    payload={"num_items": int(V.shape[0])}
                )
            ]
        )
        created += 1