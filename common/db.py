from sqlalchemy import create_engine, text
import polars as pl
import boto3
import os
from qdrant_client import QdrantClient

class Connection:
    def __init__(self):
        self.RDS_DATABASE = os.getenv("RDS_DATABASE") or os.getenv("DB_NAME")
        self.RDS_HOST = os.getenv("RDS_HOST") or os.getenv("DB_HOST")
        self.RDS_USER = os.getenv("RDS_USER") or os.getenv("DB_USER")
        self.RDS_PASSWORD = os.getenv("RDS_PASSWORD") or os.getenv("DB_PASSWORD")
        self.RDS_PORT = os.getenv("RDS_PORT") or os.getenv("DB_PORT")
        self.QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT") or os.getenv("QDRANT_URL")
        self.QDRANT_API = os.getenv("QDRANT_API") or os.getenv("QDRANT_API_KEY")

        # SSH 터널 및 DB 연결
        self.engine = None
        self._connect_to_rds()

    def _connect_to_rds(self):
        """ SSH 터널을 통해 RDS에 연결하고 SQLAlchemy 엔진 생성 """
        try:
            # SQLAlchemy 엔진 생성
            self.engine = create_engine(
                f"mysql+pymysql://{self.RDS_USER}:{self.RDS_PASSWORD}@"
                f"{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DATABASE}"
            )

        except Exception as e:
            print("Error occurred:", e)

    def execute(self, query):
        if not self.engine:
            raise Exception("No SQLAlchemy engine initialized")

        with self.engine.connect() as conn:
            return pl.read_database(query, conn)

    def _raw_execute(self, query):
        if not self.engine:
            raise Exception("No SQLAlchemy engine initialized")

        with self.engine.connect() as conn:
            conn.execute(text(query))
            if query.strip().lower().startswith(("insert", "update", "delete")):
                conn.commit()

    def save_parquet(self, query: str, path: str):
        """ SQL 결과를 parquet 파일로 저장 """
        df = self.execute(query)
        df.write_parquet(path)
        print(f"Saved to {path}")

    def get_dynamo(self):
        region = os.getenv("DYNAMODB_REGION", "ap-northeast-2")
        endpoint_url = os.getenv("DYNAMODB_ENDPOINT_URL", "").strip()

        boto3_params = {"region_name": region}
        if endpoint_url:
            boto3_params["endpoint_url"] = endpoint_url
            boto3_params["aws_access_key_id"] = os.getenv("DYNAMODB_ACCESS_KEY_ID", "dummy")
            boto3_params["aws_secret_access_key"] = os.getenv("DYNAMODB_SECRET_ACCESS_KEY", "dummy")

        try:
            _dynamo_resource = boto3.resource("dynamodb", **boto3_params)
        except Exception as e:
            print(f"Dynamo Connection Failed : {e}")
            return

        return _dynamo_resource

    def get_qdrant(self):
        client = QdrantClient(
            url=self.QDRANT_ENDPOINT,
            api_key=self.QDRANT_API
        )

        return client

    def close(self):
        """ 연결 종료 """
        if self.engine:
            self.engine.dispose()
            print("SQLAlchemy Engine Disposed...")
