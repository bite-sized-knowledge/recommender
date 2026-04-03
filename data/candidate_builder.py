import polars as pl
import numpy as np
from data.pipeline.fetch_data import *
from data.pipeline.behavior_embedding import build_behavior_embedding
from data.pipeline.initial_embedding import build_user_initial_embedding
from data.pipeline.mix_user_embedding import build_all_user_embeddings
from data.pipeline.popularity import compute_popular
from data.candidate_merge import merge_candidates_for_all_users
from typing import Any, Dict
from utils.logger import get_logger

class CandidateBuilder:
    """
    후보 필터링을 위해 데이터를 준비하고 저장하는 클래스
    """
    def __init__(
            self, 
            conn: Any,
            config: Dict[str, str],
        ):

        self.conn = conn
        self.qdrant = conn.get_qdrant()
        self.config = config
        self.logger = get_logger("Candidate Builder")

    def run(self):

        # 1. Category Centroid Embedding
        build_category_profiles(
            client=self.qdrant,
            min_points=3
        )


        # 2. User Initial Embedding = mean of picked categories
        user_init_embedding = build_user_initial_embedding(
            conn=self.conn,
            client=self.qdrant,
            dim = 512
        )

        # 3. Build user behavior embedding
        user_behavior_embedding = build_behavior_embedding(
            conn=self.conn,
            qdrant=self.qdrant,
        )

        build_all_user_embeddings(
            user_init_embedding,
            user_behavior_embedding,
            self.qdrant
        )

        popular = []

        for case in [(7,3), (30, 14)]:
            start, end = case
            popular.append(
                compute_popular(
                    self.conn,
                    start=start,
                    end=end
                )
            )

        candidates = merge_candidates_for_all_users(
            self.conn,
            self.qdrant,
            popular,
        ) 

        return candidates