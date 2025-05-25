import os
import polars as pl
import numpy as np
from data.pipeline.fetch_data import *
from data.pipeline.content_embedding import *
from embeddings.model import Embedder
from datetime import datetime, timedelta
from typing import Any, Dict
from utils.logger import get_logger

class CandidateBuilder:
    """
    후보 필터링을 위해 데이터를 준비하고 저장하는 클래스
    """
    def __init__(
            self, 
            process_dir: str,
            output_dir: str,
            conn: Any,
            config: Dict[str, str],
            train: bool = True
        ):

        self.process_dir = process_dir
        self.output_dir = output_dir
        self.conn = conn
        self.config = config
        self.logger = get_logger("Candidate Builder")
        self.train = train
        self.user_positive_map = None
        self.all_article_ids = None

        self.prepare()
 
    def prepare(self):
        """ 시간 범위 설정 """
        self.logger.info("Preparing...")
        os.makedirs(self.process_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        now = datetime.now()

        if self.train:
            start_ts = int((now - timedelta(days=7)).timestamp() * 1000)
            end_ts = int((now - timedelta(days=1)).timestamp() * 1000)
        else:
            start_ts = int((now - timedelta(days=1)).timestamp() * 1000)
            end_ts = int((now - timedelta(days=0)).timestamp() * 1000)

           
        self.user_positive_map = fetch_positive_logs(self.conn, start_ts, end_ts)
        self.all_article_ids = fetch_all_article_ids()
   

    def process_and_save(self):
        """
        데이터를 처리하고 결과를 저장합니다.
        """

        fetch_articles_to_parquet(self.conn)

        # User 및 Item 임베딩 생성
        embedder = Embedder(model_path=self.config['emb_model']['path'])
        update_embeddings(embedder)
        generate_user_embeddings(self.user_positive_map)
        generate_category_embedding(
            fetch_category_id_articles()
        )

        # 학습 데이터 생성
        dataset = generate_training_samples(self.user_positive_map)
        self._save_to_parquet(
            dataset, 
            ["member_id", "article_id", "label"],
            "training_samples.parquet"
        )

        # 인기 아티클 저장 
        popularity_articles = generate_popularity_candidates(days=14)
        self._save_to_parquet(
            popularity_articles, 
            ["article_id"],
            "popular_top_50.parquet"
        )

        # User-Item 코사인 유사도 계산
        generate_sim_candidates(
            n_rows=100,
            preferred_map=build_preferred_category_map(
                self.conn,
                self.user_positive_map
            )
        )


    def _save_to_file(self, data, file_name):
        """
        데이터를 파일에 저장합니다.
        """
        file_path = os.path.join(self.process_dir, file_name)
        np.save(file_path, data)

    def _save_to_parquet(self, data: Any, columns: List[str], file_name: str) -> None:
        """
        데이터를 Parquet 형식으로 저장합니다 (Polars 기반).

        Args:
            data (Any): 저장할 데이터 (list of dicts 또는 list of lists)
            columns (List[str]): 컬럼 이름
            file_name (str): 저장할 파일명 (예: candidates.parquet)
        """
        # data 형태에 따라 처리 분기
        file_path = os.path.join(self.process_dir, file_name)

        if data is None:
            self.logger("No Data Fetched...")
            return

        df = pl.DataFrame(data, schema=columns, orient="row")
        df.write_parquet(file_path, compression="zstd")