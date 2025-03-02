import os
from utils.db_conn import Connection
from utils.postprocess import postprocess_by_blog_id
from utils.category import CATEGORY_DICT
from llm_pipeline import LangChainModel
import warnings
import boto3
from tqdm import tqdm
warnings.filterwarnings("ignore")
conn = Connection()

def get_raw_articles():
    query = f"""
    SELECT 
        article_id, 
        content
    FROM 
        article
    """
    return conn.execute(query)


def update_article(article_id, category, keywords, length, lang):    
    keywords = '\t'.join(keywords)
    if len(keywords) == 0:
        keywords = 'NULL'

    query = f"""
    UPDATE article
    SET 
        category_id = {CATEGORY_DICT.get(category, 'NULL')},
        keywords = '{keywords}',
        content_length = {length},
        lang = '{lang}'
    WHERE 
        article_id = '{article_id}';
    """
    conn._raw_execute(query)

def run():
    raw_articles = get_raw_articles()
    model = LangChainModel()


    total_articles = raw_articles.shape[0]
    progress_bar = tqdm(total=total_articles, desc="Processing Articles")

    prediction_list = []

    for _, article in raw_articles.iterrows():
        text = article['content']
        article_id = article['article_id']
        prediction = model.predict(text)
        prediction_list.append([
            article_id, 
            prediction.focusing, 
            prediction.keywords,
            prediction.content_length,
            prediction.lang
        ])


    progress_bar = tqdm(total=len(prediction_list), desc="Updating articles' info")
    for article_id, focusing, keywords, content_length, lang in prediction_list:
        update_article(
            article_id=article_id, 
            category=focusing, 
            keywords=keywords,
            length=content_length,
            lang=lang
        )
        progress_bar.update(1)
    
    print("LLM Classification Done..!")
        

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = boto3.client("ssm", region_name="ap-northeast-2").\
                                        get_parameter(Name='/llm/apikey', WithDecryption=True)\
                                        ["Parameter"]["Value"]

    run()
    conn.close()