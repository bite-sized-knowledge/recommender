import boto3
import pymysql
from sshtunnel import SSHTunnelForwarder
import pandas as pd

class Connection:
    def __init__(self):
        self.RDS_DATABASE = "bite-dev"
        self.RDS_HOST = self._get_parameter("/rds/db-host")
        self.RDS_USER = self._get_parameter("/rds/db-user")
        self.RDS_PASSWORD = self._get_parameter("/rds/db-password")
        self.RDS_PORT = 3306

        self.SSH_HOST = self._get_parameter("/rds/ec2-host")
        self.SSH_KEY_FILE = "/tmp/temp.pem"

        # SSH 터널 및 DB 연결
        self._connect_to_rds()

    def _get_parameter(self, name):
        """ AWS SSM Parameter Store에서 값 가져오기 """
        client = boto3.client("ssm", region_name="ap-northeast-2")
        response = client.get_parameter(Name=name, WithDecryption=True)
        return response["Parameter"]["Value"]

    def _connect_to_rds(self):
        """ SSH 터널을 통해 RDS에 연결 """
        try:
            self.tunnel = SSHTunnelForwarder(
                (self.SSH_HOST, 22),
                ssh_username="ec2-user",
                ssh_pkey=self.SSH_KEY_FILE,
                remote_bind_address=(self.RDS_HOST, self.RDS_PORT),
            )
            self.tunnel.start()
            print(f"SSH Connected! {self.tunnel.local_bind_address} -> RDS_HOST:RDS_PORT")

            # MySQL 연결
            self.connection = pymysql.connect(
                host="localhost",
                user=self.RDS_USER,
                password=self.RDS_PASSWORD,
                database=self.RDS_DATABASE,
                port=self.tunnel.local_bind_port,
            )
            print("Connected to RDS!")

        except Exception as e:
            print("Error occured:", e)

    def execute(self, query):
        if not self.connection:
            raise Exception("There is nothing connection made")

        return pd.read_sql(query, self.connection)

    def _raw_execute(self, query):
        if not self.connection:
            raise Exception("There is nothing connection made")

        cursor = self.connection.cursor()
        cursor.execute(query)
        self.connection.commit()
        cursor.close()

    def close(self):
        """ 연결 종료 """
        if self.connection:
            self.connection.close()
            print("RDS Connection Closing...")
        if self.tunnel:
            self.tunnel.stop()
            print("SSH Tunneling Closing...")