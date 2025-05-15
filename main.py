import os
import logging
from data.pipeline.fetch_data import *
from common.db import *
from utils.logger import get_logger
from utils.config_loader import load_config
from embeddings.model import Embedder
from dotenv import load_dotenv

logging.getLogger("gensim.models.utils_any2utf8").setLevel(logging.CRITICAL)
logger = get_logger("Candidate-Filtering")
load_dotenv()


def run():
    try:
        conn = Connection()
        config = load_config()

        logger.info("=== Fetch Start ===")

        fetch_articles_to_parquet(conn)
        fetch_events_to_parquet(conn)

        embedder = Embedder(config['emb_model']['path'])

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    run()