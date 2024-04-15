

'''
import json
from pymysql import Connection
from kafka import KafkaProducer

def fetch_data_from_mysql(host, port, user, password, database, table):
    """从MySQL数据库中获取数据"""
    con = mysql.connector.connect(
        ost='localhost',  # 主机名（IP）
        port=3306,  # 端口
        user='root',  # 账户名
        password='@Fantasy2024',  # 密码
    )

    cursor = con.cursor(dictionary=True)
    cursor.execute("SELECT * FROM {}".format(table))
    rows = cursor.fetchall()

    cursor.close()
    con.close()

    return rows

def produce_to_kafka(bootstrap_servers, topic, data):
    """将数据发送到Kafka"""
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    for row in data:
        producer.send(topic, value=row)

    producer.flush()
    producer.close()

if __name__ == "__main__":
    # MySQL数据库连接信息
    mysql_host = 'localhost'
    mysql_port = 3306
    mysql_user = 'your_mysql_user'
    mysql_password = 'your_mysql_password'
    mysql_database = 'your_mysql_database'
    mysql_table = 'your_mysql_table'

    # Kafka服务器连接信息
    kafka_bootstrap_servers = 'localhost:9092'
    kafka_topic = 'your_kafka_topic'

    # 从MySQL中获取数据
    data_from_mysql = fetch_data_from_mysql(mysql_host, mysql_port, mysql_user, mysql_password, mysql_database, mysql_table)

    # 将数据发送到Kafka
    produce_to_kafka(kafka_bootstrap_servers, kafka_topic, data_from_mysql)
'''