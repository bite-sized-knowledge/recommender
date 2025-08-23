from data.pipeline.fetch_data import *
from data.candidate_builder import CandidateBuilder
# from data.candidate_merge import merge_candidates
from data.seg_routing import UserSegmentation
from common.db import Connection
from utils.logger import get_logger
from utils.config_loader import load_config
from utils.candidate_save import save_recommendations_to_db

# 로깅 설정
logger = get_logger("Candidate-Filtering")


def run_pipeline():
    """
    추천 시스템 파이프라인을 실행합니다.
    """
    try:
        logger.info("=== Connection & Config loading... ===")
        conn = Connection()
        config = load_config()

        logger.info("=== User Segment Routing... ===")
        router = UserSegmentation(conn)
        segments = router.run()

        logger.info("=== Candiate Building... === ")
        builder = CandidateBuilder(
            conn,
            config
        )
        candidates = builder.run()

        logger.info("=== Saving Data... ===")
        save_recommendations_to_db(
            candidates,
            conn
        )



    except Exception as e:
        logger.exception(f"Pipeline failed : {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    run_pipeline()