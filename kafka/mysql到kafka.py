import json
from pymysql import Connection
from kafka import KafkaProducer

def fetch_data_from_mysql(host, port, user, password, database, table):
    """从MySQL数据库中获取数据"""
    # 建立与MySQL数据库的连接
    con = mysql.connector.connect(
        host=host,       # 主机名（IP）
        port=port,       # 端口
        user=user,       # 账户名
        password=password,  # 密码
        database=database    # 数据库名称
    )

    # 创建游标对象
    cursor = con.cursor(dictionary=True)

    # 执行SQL查询
    cursor.execute("SELECT * FROM {}".format(table))

    # 获取所有查询结果
    rows = cursor.fetchall()

    # 关闭游标和数据库连接
    cursor.close()
    con.close()

    return rows

def produce_to_kafka(bootstrap_servers, topic, data):
    """将数据发送到Kafka"""
    # 创建Kafka生产者对象
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    # 遍历数据并发送到指定的Kafka主题
    for row in data:
        producer.send(topic, value=row)

    # 刷新并关闭生产者
    producer.flush()
    producer.close()

if __name__ == "__main__":
    # MySQL数据库连接信息
    mysql_host = 'localhost'          # MySQL主机名
    mysql_port = 3306                 # MySQL端口
    mysql_user = 'your_mysql_user'    # MySQL用户名
    mysql_password = 'your_mysql_password'  # MySQL密码
    mysql_database = 'your_mysql_database'  # MySQL数据库名
    mysql_table = 'your_mysql_table'        # MySQL表名

    # Kafka服务器连接信息
    kafka_bootstrap_servers = 'localhost:9092'  # Kafka服务器地址及端口
    kafka_topic = 'your_kafka_topic'            # Kafka主题

    # 从MySQL中获取数据
    data_from_mysql = fetch_data_from_mysql(mysql_host, mysql_port, mysql_user, mysql_password, mysql_database, mysql_table)

    # 将数据发送到Kafka
    produce_to_kafka(kafka_bootstrap_servers, kafka_topic, data_from_mysql)
