import os
import ssl

import polars as pl
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, event, text


# engagement_aggregator 의 MAX(CASE WHEN ... THEN occurred_at END) 가 빈 group 또는 invalid datetime 를
# 만나면 NO_ZERO_DATE sql_mode 에서 cast reject. 배치는 운영 데이터 관용적으로 처리해야 하므로
# session-level 에서 NO_ZERO_DATE / NO_ZERO_IN_DATE 만 제거 (다른 strict mode 는 유지).
_BATCH_SQL_MODE = (
    "STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION"
)


class Connection:
    def __init__(self):
        self.RDS_DATABASE = os.getenv("RDS_DATABASE") or os.getenv("DB_NAME")
        self.RDS_HOST = os.getenv("RDS_HOST") or os.getenv("DB_HOST")
        self.RDS_USER = os.getenv("RDS_USER") or os.getenv("DB_USER")
        self.RDS_PASSWORD = os.getenv("RDS_PASSWORD") or os.getenv("DB_PASSWORD")
        self.RDS_PORT = os.getenv("RDS_PORT") or os.getenv("DB_PORT")
        self.MYSQL_CA_PATH = os.getenv("MYSQL_CA_PATH") or os.getenv("DB_TLS_CA")
        self.QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT") or os.getenv("QDRANT_URL")
        self.QDRANT_API = os.getenv("QDRANT_API") or os.getenv("QDRANT_API_KEY")

        self.engine = None
        self._connect_to_rds()

    def _connect_to_rds(self):
        """MySQL 8.0 require_secure_transport=ON 이라 SSL context 필수.
        CA 가 마운트돼있으면 검증, 없으면 self-signed 무시 (recsys-serving 패턴 동일)."""
        if self.MYSQL_CA_PATH and os.path.exists(self.MYSQL_CA_PATH):
            ssl_ctx = ssl.create_default_context(cafile=self.MYSQL_CA_PATH)
            ssl_ctx.check_hostname = False  # self-signed CN doesn't match Docker service name
        else:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        self.engine = create_engine(
            f"mysql+pymysql://{self.RDS_USER}:{self.RDS_PASSWORD}@"
            f"{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DATABASE}",
            pool_pre_ping=True,
            connect_args={"ssl": ssl_ctx},
        )

        @event.listens_for(self.engine, "connect")
        def _set_session_sql_mode(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            try:
                cur.execute(f"SET SESSION sql_mode = '{_BATCH_SQL_MODE}'")
            finally:
                cur.close()

    def execute(self, query):
        if not self.engine:
            raise Exception("No SQLAlchemy engine initialized")

        with self.engine.connect() as conn:
            return pl.read_database(query, conn)

    def _raw_execute(self, query, params=None):
        if not self.engine:
            raise Exception("No SQLAlchemy engine initialized")

        with self.engine.connect() as conn:
            conn.execute(text(query), params or {})
            if query.strip().lower().startswith(("insert", "update", "delete")):
                conn.commit()

    def _batch_execute(self, query, param_list):
        """Execute a parameterized query with all params in one executemany call."""
        if not self.engine:
            raise Exception("No SQLAlchemy engine initialized")
        if not param_list:
            return

        with self.engine.connect() as conn:
            conn.execute(text(query), param_list)
            conn.commit()

    def save_parquet(self, query: str, path: str):
        """ SQL 결과를 parquet 파일로 저장 """
        df = self.execute(query)
        df.write_parquet(path)
        print(f"Saved to {path}")

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
