"""Shared constants & helpers for recommender pipelines."""
import uuid

import numpy as np

# Qdrant collections
ITEM_COLLECTION = "bite-vectordb"
USER_COLLECTION = "user_profile"

UUID_NAMESPACE = uuid.NAMESPACE_DNS


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (v / n).astype(np.float32)


def _to_point_id(article_id: str, namespace=UUID_NAMESPACE) -> uuid.UUID:
    return uuid.uuid5(namespace, article_id)
