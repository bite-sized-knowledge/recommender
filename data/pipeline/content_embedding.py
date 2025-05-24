import polars as pl
from pathlib import Path
from utils.logger import get_logger


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
    for article_id, sentence in zip(df["article_id"].to_list(), texts):
        vec = embedder.get_vector_from_text(sentence)
        records.append({
            "article_id": article_id,
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