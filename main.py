from dotenv import load_dotenv
from data.pipeline.fetch_data import *
from data.builder import CandidateBuilder
from common.db import Connection
from utils.logger import get_logger
from utils.config_loader import load_config

# 로깅 설정
logger = get_logger("Candidate-Filtering")
load_dotenv()

def run_pipeline():
    """
    추천 시스템 파이프라인을 실행합니다.
    """
    try:
        logger.info("=== Connection & Config loading... ===")
        conn = Connection()
        config = load_config()

        logger.info("=== Candiate Building... === ")
        builder = CandidateBuilder("./data/processed",conn,config)
        builder.process_and_save()



    except Exception as e:
        logger.exception(f"Pipeline failed : {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    run_pipeline()