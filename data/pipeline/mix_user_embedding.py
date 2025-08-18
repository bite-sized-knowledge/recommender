import numpy as np
import polars as pl
from typing import Optional
from data.utils import (
    _now_ms, EVENT_WEIGHTS
)
from qdrant_client.models import (
    PointStruct,
)

USER_COLLECTION = "user-profiles"


def _effective_samples(
    user_logs: pl.DataFrame,
    *,
    now_ms: int,
    half_life_days: float = 3.0,
    event_weights: dict[str, float] = EVENT_WEIGHTS,
    diversity_bonus: float = 0.1,   # 이벤트 타입 1개 늘 때마다 보너스 비율
) -> float:
    """유저 로그로부터 유효 샘플 수 n_eff 계산."""
    if user_logs.is_empty():
        return 0.0

    # 시간 감쇠 계수: 0.5 ** (delta_days / HL)
    HL = half_life_days

    df = user_logs.select(
        pl.col("event_type"),
        pl.col("timestamp"),
        ((now_ms - pl.col("timestamp")) / (1000 * 60 * 60 * 24)).alias("delta_days")
    ).with_columns(
        # 시간 감쇠
        (0.5 ** (pl.col("delta_days") / HL)).alias("time_decay"),
        # 이벤트 가중치 매핑
        pl.col("event_type").replace(event_weights, default=0.0).alias("evt_w")
    )

    # 기본 유효 샘플 합
    n_eff_base = (df["time_decay"] * df["evt_w"]).sum()

    # 행동 다양성 보너스: 타입이 다양하면 신뢰도 가중
    n_types = user_logs["event_type"].n_unique()
    diversity_factor = 1.0 + diversity_bonus * max(0, n_types - 1)

    return float(n_eff_base * diversity_factor)

def compute_alpha(
    user_logs: pl.DataFrame,
    *,
    now_ms: int,
    k: float = 10.0,               # prior 강도 (클수록 alpha가 천천히 올라감)
    max_alpha: float = 0.85,       # 행동 가중치 상한
    **kwargs                        # half_life_days, event_weights, diversity_bonus 전달용
) -> float:
    n_eff = _effective_samples(user_logs, now_ms=now_ms, **kwargs)
    if n_eff <= 0:
        return 0.0
    alpha = n_eff / (n_eff + k)
    return float(min(alpha, max_alpha))

def build_user_embedding(
        cat_vec: Optional[np.ndarray],
        beh_vec: Optional[np.ndarray],
        user_logs_pl: pl.DataFrame,
        now_ms: int
    ) -> Optional[np.ndarray]:
    """
    category + behavior + logs 기반 최종 유저 임베딩 생성
    """
    # behavior가 없거나 로그가 비었으면 → category만
    if beh_vec is None or user_logs_pl.is_empty():
        return 0, cat_vec

    # category가 없으면 → behavior만
    if cat_vec is None:
        return 0, beh_vec

    # logs로 alpha 산출
    alpha = compute_alpha(
        user_logs_pl,
        now_ms=now_ms,
        k=5.0,
        half_life_days=3.0,
        diversity_bonus=0.1,
        event_weights=EVENT_WEIGHTS,
        max_alpha=0.85
    )
    return alpha, (1 - alpha) * cat_vec + alpha * beh_vec

def build_all_user_embeddings(
        cat_vec,
        beh_vec,
        client
    ):
    """
    category dict + behavior dict를 결합해 최종 user embedding 생성
    최종 형태: {user_id: vector(list[float])}
    Qdrant Upsert
    """
    now_ms = _now_ms()


    # Mixing
    uids = set(cat_vec.keys()) | set(beh_vec.keys())
    points = []

    for uid in uids:
        c_vec = cat_vec.get(uid)
        beh_entry = beh_vec.get(uid, {"vector": None, "logs": pl.DataFrame()})
        b_vec = beh_entry["vector"]
        pl_logs = beh_entry["logs"] if isinstance(beh_entry["logs"], pl.DataFrame) else pl.DataFrame()

        alpha, weighted = build_user_embedding(c_vec, b_vec, pl_logs, now_ms)
        if weighted is not None:
            points.append(
                PointStruct(
                    id=uid,
                    vector=weighted.tolist(),
                    payload={"alpha" : alpha}
                )

            )

    client.upsert(collection_name=USER_COLLECTION, points=points)
    print("[Weighted Hybrid Embedding] Calculation Complete & Upsert into Qdrant...") 