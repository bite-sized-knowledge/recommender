import time

from data.candidate_builder import CandidateBuilder
from data.seg_routing import UserSegmentation
from common.db import Connection
from utils.logger import get_logger
from utils.config_loader import load_config
from utils.candidate_save import save_recommendations_to_db
from utils.metric_sink import MetricSink
from data.pipeline.engagement_aggregator import aggregate_engagement
from data.pipeline.user_features import compute_user_features, update_user_profile_payloads
from data.pipeline.article_metrics import compute_article_metrics
from ranker.scoring import rank_candidates

logger = get_logger("Candidate-Filtering")


def run_pipeline():
    """
    추천 시스템 파이프라인을 실행합니다.
    """
    pipeline_start = time.time()
    metrics = {}
    sink = None
    error = None
    conn = None

    try:
        logger.info("=== Connection & Config loading... ===")
        conn = Connection()
        config = load_config()
        sink = MetricSink(conn)
        logger.info(f"=== Run id: {sink.run_id} ===")

        logger.info("=== User Segment Routing... ===")
        t0 = time.time()
        router = UserSegmentation(conn)
        segments = router.run()
        stage_ms = int((time.time() - t0) * 1000)
        metrics["segmentation_ms"] = stage_ms

        seg_payload = {}
        if not segments.is_empty():
            seg_counts = segments.group_by("user_segment").len()
            for row in seg_counts.iter_rows(named=True):
                key = str(row["user_segment"])
                seg_payload[key] = int(row["len"])
                metrics[f"seg_{key}"] = int(row["len"])
            logger.info(f"Segments: {seg_counts.to_dicts()}")
        sink.record("segmentation", stage_ms, seg_payload or None)

        logger.info("=== Candidate Building... ===")
        t0 = time.time()
        builder = CandidateBuilder(
            conn,
            config,
            segments=segments,
        )
        candidates = builder.run()
        stage_ms = int((time.time() - t0) * 1000)
        metrics["candidate_build_ms"] = stage_ms
        metrics["candidates_total"] = len(candidates)

        candidate_payload = {"candidates_total": len(candidates)}
        if not candidates.is_empty():
            per_user = candidates.group_by("member_id").len()
            avg = round(per_user["len"].mean(), 1)
            mn = int(per_user["len"].min())
            mx = int(per_user["len"].max())
            metrics["candidates_per_user_avg"] = avg
            metrics["candidates_per_user_min"] = mn
            metrics["candidates_per_user_max"] = mx
            candidate_payload.update(
                {
                    "candidates_per_user_avg": avg,
                    "candidates_per_user_min": mn,
                    "candidates_per_user_max": mx,
                }
            )
        sink.record("candidate_build", stage_ms, candidate_payload)

        logger.info("=== Ranking Candidates... ===")
        t0 = time.time()
        ranked = rank_candidates(candidates, conn, conn.get_qdrant())
        stage_ms = int((time.time() - t0) * 1000)
        metrics["ranking_ms"] = stage_ms
        metrics["ranked_total"] = len(ranked)

        rank_payload = {"ranked_total": len(ranked)}
        if not ranked.is_empty():
            mn = round(ranked["score"].min(), 4)
            mx = round(ranked["score"].max(), 4)
            mean = round(ranked["score"].mean(), 4)
            metrics["score_min"] = mn
            metrics["score_max"] = mx
            metrics["score_mean"] = mean
            rank_payload.update({"score_min": mn, "score_max": mx, "score_mean": mean})
        sink.record("ranking", stage_ms, rank_payload)

        logger.info("=== Saving Data... ===")
        t0 = time.time()
        save_recommendations_to_db(
            ranked,
            conn
        )
        stage_ms = int((time.time() - t0) * 1000)
        metrics["save_ms"] = stage_ms
        sink.record("save", stage_ms)

        logger.info("=== Aggregating Engagement Scores... ===")
        t0 = time.time()
        aggregate_engagement(conn)
        stage_ms = int((time.time() - t0) * 1000)
        metrics["engagement_ms"] = stage_ms
        sink.record("engagement", stage_ms)

        logger.info("=== Computing User Features... ===")
        t0 = time.time()
        user_feats = compute_user_features(conn)
        if user_feats:
            update_user_profile_payloads(conn.get_qdrant(), user_feats)
        stage_ms = int((time.time() - t0) * 1000)
        metrics["user_features_ms"] = stage_ms
        metrics["users_with_features"] = len(user_feats)
        sink.record(
            "user_features",
            stage_ms,
            {"users_with_features": len(user_feats)},
        )

        logger.info("=== Computing Article Metrics... ===")
        t0 = time.time()
        article_metrics = compute_article_metrics(conn)
        stage_ms = int((time.time() - t0) * 1000)
        metrics["article_metrics_ms"] = stage_ms
        metrics["articles_with_metrics"] = len(article_metrics)
        sink.record(
            "article_metrics",
            stage_ms,
            {"articles_with_metrics": len(article_metrics)},
        )

    except Exception as e:
        error = e
        logger.exception(f"Pipeline failed : {e}")
        metrics["error"] = str(e)

    finally:
        total_ms = int((time.time() - pipeline_start) * 1000)
        metrics["pipeline_total_ms"] = total_ms

        if sink is not None:
            payload = None
            if error is not None:
                payload = {
                    "error_class": type(error).__name__,
                    "error": str(error),
                }
            try:
                sink.record("pipeline_total", total_ms, payload)
            except Exception:
                logger.debug("pipeline_total record skipped", exc_info=True)

        logger.info(f"=== Pipeline Metrics === {metrics}")

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    run_pipeline()
