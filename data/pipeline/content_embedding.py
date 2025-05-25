import polars as pl
import numpy as np
from pathlib import Path
from utils.logger import get_logger
from typing import List, Dict, Set, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity


def update_embeddings(embedder):
    """
    Update(or generate) embedding of all articles
    using title+description+keywords
    """

    DATA_PATH = Path("data/raw/articles.parquet")
    EMBED_PATH = Path("data/processed/item_embeddings.parquet")

    print()
    logger = get_logger("Update Embeddings")
    logger.info("Loading article data...")
    df = pl.read_parquet(DATA_PATH)

    logger.info("Checking for existing embeddings...")
    if EMBED_PATH.exists():
        existing_df = pl.read_parquet(EMBED_PATH)
        existing_ids = set(existing_df["article_id"].to_list())
        df = df.filter(~pl.col("article_id").is_in(existing_ids))
    else:
        existing_df = None

    logger.info(f"New articles to embed: {df.shape[0]}")
    if df.is_empty():
        logger.info("No new articles to embed.")
        return

    logger.info("Preparing sentences for model update...")
    df = df.with_columns([
        (pl.col("title") + " " + pl.col("description") + " " + pl.col("keywords")).alias("full_text")
    ])
    texts = df["full_text"].to_list()

    logger.info("Generating new embeddings...")
    records = []
    for article_id, category_id, sentence in zip(df["article_id"].to_list(), df["category_id"], texts):
        vec = embedder.get_vector_from_text(sentence)
        records.append({
            "article_id": article_id,
            "category_id": category_id,
            **{f"dim_{i}": vec[i] for i in range(len(vec))}
        })

    new_df = pl.DataFrame(records)

    logger.info("Merging and saving embeddings...")
    if existing_df is not None:
        merged = pl.concat([existing_df, new_df], how="vertical")
    else:
        merged = new_df

    merged.write_parquet(EMBED_PATH, compression="zstd")
    logger.info("Embedding update complete.")
    

def generate_user_embeddings(user_article_map):
    ITEM_EMBED_PATH = Path("data/processed/item_embeddings.parquet")
    USER_EMBED_PATH = Path("data/processed/user_embeddings.parquet")

    print()
    logger = get_logger("Generating User Embeddings")

    logger.info("Loading item embeddings...")
    item_embeds = pl.read_parquet(ITEM_EMBED_PATH)

    article_to_vector = {
        row["article_id"]: row for row in item_embeds.iter_rows(named=True)
    }

    embed_cols = [col for col in item_embeds.columns if col.startswith("dim_")]
    user_vectors = []

    logger.info("Computing mean embeddings per user...")
    for user_id, article_ids in user_article_map.items():
        vectors = [
            [article_to_vector[aid][col] for col in embed_cols]
            for aid in article_ids
            if aid in article_to_vector
        ]
        if not vectors:
            continue

        mean_vector = [sum(col) / len(col) for col in zip(*vectors)]
        user_vectors.append({"member_id": user_id, **dict(zip(embed_cols, mean_vector))})
    
    logger.info(f"Saving {len(user_vectors)} user embeddings...")
    pl.DataFrame(user_vectors).write_parquet(USER_EMBED_PATH, compression="zstd")
    logger.info("Done. User embeddings saved.")
    print()


def generate_category_embedding(category_article_map):
    ITEM_EMBED_PATH = Path("data/processed/item_embeddings.parquet")
    CATEGORY_EMBED_PATH = Path("data/processed/category_embeddings.parquet")

    print()
    logger = get_logger("Generating Category_id Embeddings")

    logger.info("Loading item embeddings...")
    item_embeds = pl.read_parquet(ITEM_EMBED_PATH)

    article_to_vector = {
        row["article_id"]: row for row in item_embeds.iter_rows(named=True)
    }

    embed_cols = [col for col in item_embeds.columns if col.startswith("dim_")]
    category_id_vector = []

    logger.info("Computing mean embeddings per category_id...")
    for category_id, article_ids in category_article_map.items():
        vectors = [
            [article_to_vector[aid][col] for col in embed_cols]
            for aid in article_ids
            if aid in article_to_vector
        ]
        if not vectors:
            continue

        mean_vector = [sum(col) / len(col) for col in zip(*vectors)]
        category_id_vector.append({"category_id": category_id, **dict(zip(embed_cols, mean_vector))})
    
    logger.info(f"Saving category_id embeddings...")
    pl.DataFrame(category_id_vector).write_parquet(CATEGORY_EMBED_PATH, compression="zstd")

def generate_sim_candidates(
    n_rows: int,
    preferred_map: Dict[int, Set[int]],
    boost_factor: float = 1.2
):

    USER_EMBED_PATH = Path("data/processed/user_embeddings.parquet")
    ITEM_EMBED_PATH = Path("data/processed/item_embeddings.parquet")
    CANDIDATE_PATH = Path("data/candidates/user_item_sim.parquet")
    logger = get_logger("Calculate User-Item Cosine Similarity")


    print()
    logger.info("Loading embeddings...")
    users = pl.read_parquet(USER_EMBED_PATH)
    items = pl.read_parquet(ITEM_EMBED_PATH)

    if "category_id" not in items.columns:
        raise ValueError("item_embeddings.parquet 파일에 'category_id' 컬럼이 필요합니다.")

    user_ids = users["member_id"].to_list()
    article_ids = items["article_id"].to_list()
    item_categories = items["category_id"].to_list()

    user_vectors = users.select([col for col in users.columns if col.startswith("dim_")]).to_numpy()
    item_vectors = items.select([col for col in items.columns if col.startswith("dim_")]).to_numpy()

    logger.info("Calculating cosine similarity...")
    sims = cosine_similarity(user_vectors, item_vectors)  # (num_users, num_items)

    logger.info("Applying preference-based boost...")
    for user_idx, member_id in enumerate(user_ids):
        preferred_categories = preferred_map.get(member_id, set())
        for item_idx, category_id in enumerate(item_categories):
            if category_id in preferred_categories:
                sims[user_idx][item_idx] *= boost_factor

    logger.info("Sorting top-K for each user...")
    top_k_indices = np.argsort(-sims, axis=1)[:, :n_rows]  # 상위 K개 인덱스

    logger.info("Building candidate list...")
    rows = []
    for user_idx, member_id in enumerate(user_ids):
        for rank, item_idx in enumerate(top_k_indices[user_idx]):
            rows.append({
                "member_id": member_id,
                "article_id": article_ids[item_idx],
                "score": float(sims[user_idx][item_idx]),
                "source": "embedding_sim",
                "rank": rank + 1
            })

    pl.DataFrame(rows).write_parquet(CANDIDATE_PATH, compression="zstd")