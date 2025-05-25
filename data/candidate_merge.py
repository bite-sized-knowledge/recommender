import polars as pl
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger("MergeCandidates")

def merge_candidates(
    candidate_dir: str = "data/candidates",
    source_weights: Dict[str, float] = {
        "embedding_sim": 0.7,
        "user_item_sim": 0.7,
        "popular": 0.3,
        "personal_popular": 0.3
    }
) -> List[Dict]:
    """
    candidates 디렉토리 내 모든 parquet 파일을 불러와 후보군 통합 및 Top-K 생성 (정규화 + 가중치 방식)

    Returns:
        List[Dict]: 통합된 후보군 리스트
    """
    logger.info("Merging candidates from directory...")
    path = Path(candidate_dir)
    files = list(path.glob("*.parquet"))

    merged_path = path / "merged.parquet"
    if merged_path in files:
        files.remove(merged_path)

    if not files:
        logger.warning(f"No candidate files found in {candidate_dir}")
        return []

    logger.info(f"Found {len(files)} candidate files.")
    grouped_by_user: Dict[int, Dict[str, Dict]] = defaultdict(dict)

    # 1. 파일 단위 정규화 + 가중치 적용
    for file in files:
        logger.info(f"Reading file: {file.name}")
        df = pl.read_parquet(file)

        source_name = df[0, "source"]
        weight = source_weights.get(source_name, 0.0)
        if weight == 0.0:
            logger.warning(f"Source '{source_name}' not found in source_weights. Skipping.")
            continue

        scores = df["score"]
        min_score, max_score = scores.min(), scores.max()

        if min_score == max_score:
            normalized_scores = pl.Series([0.7] * len(scores))
        else:
            normalized_scores = (scores - min_score) / (max_score - min_score)

        final_scores = normalized_scores * weight
        df = df.with_columns([
            final_scores.alias("final_score")
        ])

        for row in df.iter_rows(named=True):
            user = row["member_id"]
            article = row["article_id"]
            final_score = row["final_score"]

            if article not in grouped_by_user[user] or final_score > grouped_by_user[user][article]["final_score"]:
                grouped_by_user[user][article] = row

    # 2. 유저별 Top-K 추출
    logger.info("Selecting Top-K candidates per user...")
    final_rows = []
    for member_id, article_dict in grouped_by_user.items():
        top_items = sorted(article_dict.values(), key=lambda x: x["final_score"], reverse=True)
        for new_rank, row in enumerate(top_items, start=1):
            final_rows.append({
                "member_id": member_id,
                "article_id": row["article_id"],
                "score": row["final_score"],  # normalized + weighted score
                "source": row["source"],
                "rank": new_rank
            })

    logger.info(f"Merged final candidates for {len(grouped_by_user)} users. Total rows: {len(final_rows)}")

    output_path = f"{candidate_dir}/merged.parquet"
    pl.DataFrame(final_rows).write_parquet(output_path, compression="zstd")
    logger.info(f"Saved merged candidates to {output_path}")
    return final_rows
