import os
from utils.db_conn import Connection
from utils.postprocess import postprocess_by_blog_id
from utils.category import category_to_idx
from llm_pipeline import LangChainModel
import warnings
import boto3
from tqdm import tqdm
warnings.filterwarnings("ignore")
conn = Connection()

def get_raw_articles():
    query = f"""
    SELECT 
        a.article_id, 
        a.blog_id,
        a.title, 
        a.content,
        a.keywords
    FROM 
        article AS a
    """
    return conn.execute(query)


def update_article(article_id, category, keywords, content, length, lang):    
    keywords = '\t'.join(keywords)
    query = f"""
    UPDATE article
    SET category_id ={category_to_idx(category)},
        keywords = '{keywords}',
        content = '{content}',
        content_length = {length},
        lang = '{lang}'
    WHERE article_id = '{article_id}';
    """
    conn._raw_execute(query)

def run():
    raw_articles = get_raw_articles()
    model = LangChainModel()
    for _, row in tqdm(raw_articles.iterrows(), total=raw_articles.shape[0], desc="Processing..."):
        article_id = row["article_id"]
        blog_id = row["blog_id"]
        content = row["content"]
        keywords = row["keywords"]
        if keywords is not None:
            continue
        postprocessed_content = postprocess_by_blog_id(content, blog_id)

        print(f"Predicting article ID: {article_id}...")
        prediction = model.predict(postprocessed_content)
        update_article(
            article_id=article_id, 
            category=prediction.focusing, 
            keywords=prediction.keywords,
            content=postprocessed_content,
            length=prediction.content_length,
            lang=prediction.lang
        )

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = boto3.client("ssm", region_name="ap-northeast-2").\
                                        get_parameter(Name='/llm/apikey', WithDecryption=True)\
                                        ["Parameter"]["Value"]

    run()
    conn.close()