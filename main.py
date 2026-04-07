import time
from data.candidate_builder import CandidateBuilder
from data.seg_routing import UserSegmentation
from common.db import Connection
from utils.logger import get_logger
from utils.config_loader import load_config
from utils.candidate_save import save_recommendations_to_db
from data.pipeline.engagement_aggregator import aggregate_engagement
from data.pipeline.user_features import compute_user_features, update_user_profile_payloads
from data.pipeline.article_metrics import compute_article_metrics
from ranker.scoring import rank_candidates

# 로깅 설정
logger = get_logger("Candidate-Filtering")


def run_pipeline():
    """
    추천 시스템 파이프라인을 실행합니다.
    """
    pipeline_start = time.time()
    metrics = {}

    try:
        logger.info("=== Connection & Config loading... ===")
        conn = Connection()
        config = load_config()

        logger.info("=== User Segment Routing... ===")
        t0 = time.time()
        router = UserSegmentation(conn)
        segments = router.run()
        metrics["segmentation_ms"] = int((time.time() - t0) * 1000)

        # Log segment distribution
        if not segments.is_empty():
            seg_counts = segments.group_by("user_segment").len()
            for row in seg_counts.iter_rows(named=True):
                metrics[f"seg_{row['user_segment']}"] = row["len"]
            logger.info(f"Segments: {seg_counts.to_dicts()}")

        logger.info("=== Candidate Building... ===")
        t0 = time.time()
        builder = CandidateBuilder(
            conn,
            config,
            segments=segments,
        )
        candidates = builder.run()
        metrics["candidate_build_ms"] = int((time.time() - t0) * 1000)
        metrics["candidates_total"] = len(candidates)

        if not candidates.is_empty():
            per_user = candidates.group_by("member_id").len()
            metrics["candidates_per_user_avg"] = round(per_user["len"].mean(), 1)
            metrics["candidates_per_user_min"] = per_user["len"].min()
            metrics["candidates_per_user_max"] = per_user["len"].max()

        logger.info("=== Ranking Candidates... ===")
        t0 = time.time()
        ranked = rank_candidates(candidates, conn, conn.get_qdrant())
        metrics["ranking_ms"] = int((time.time() - t0) * 1000)
        metrics["ranked_total"] = len(ranked)

        if not ranked.is_empty():
            metrics["score_min"] = round(ranked["score"].min(), 4)
            metrics["score_max"] = round(ranked["score"].max(), 4)
            metrics["score_mean"] = round(ranked["score"].mean(), 4)

        logger.info("=== Saving Data... ===")
        t0 = time.time()
        save_recommendations_to_db(
            ranked,
            conn
        )
        metrics["save_ms"] = int((time.time() - t0) * 1000)

        logger.info("=== Aggregating Engagement Scores... ===")
        t0 = time.time()
        aggregate_engagement(conn)
        metrics["engagement_ms"] = int((time.time() - t0) * 1000)

        logger.info("=== Computing User Features... ===")
        t0 = time.time()
        user_feats = compute_user_features(conn)
        if user_feats:
            update_user_profile_payloads(conn.get_qdrant(), user_feats)
        metrics["user_features_ms"] = int((time.time() - t0) * 1000)
        metrics["users_with_features"] = len(user_feats)

        logger.info("=== Computing Article Metrics... ===")
        t0 = time.time()
        article_metrics = compute_article_metrics(conn)
        metrics["article_metrics_ms"] = int((time.time() - t0) * 1000)
        metrics["articles_with_metrics"] = len(article_metrics)

    except Exception as e:
        logger.exception(f"Pipeline failed : {e}")
        metrics["error"] = str(e)

    finally:
        try:
            conn.close()
        except Exception:
            pass

    metrics["pipeline_total_ms"] = int((time.time() - pipeline_start) * 1000)
    logger.info(f"=== Pipeline Metrics === {metrics}")

if __name__ == "__main__":
    run_pipeline()
