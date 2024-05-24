import pymysql
from kafka import KafkaProducer
import json

# MySQL配置
mysql_host = 'localhost'  # MySQL服务器地址
mysql_user = 'root'  # MySQL用户名
mysql_password = '@Fantasy2024'  # MySQL密码
mysql_db = 'test_db'  # 要连接的数据库名称
mysql_table = 'test_table'  # 要查询的表名称

# Kafka配置
bootstrap_servers = ['localhost:9092']  # Kafka服务器地址
topic_name = 'test_topic'  # Kafka主题名称

# 创建MySQL连接
conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, db=mysql_db)

# 创建游标
cursor = conn.cursor()

# 查询MySQL数据
cursor.execute(f"SELECT * FROM {mysql_table}")

# 创建Kafka生产者
producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# 发送消息
for row in cursor:
    message = {'id': row[0], 'data': row[1]}  # 将每一行数据转换为字典
    producer.send(topic_name, value=message)  # 发送消息到Kafka主题
    print(f"发送消息的: {message}")  # 打印已发送的消息

# 关闭生产者
producer.flush()  # 刷新所有未发送的消息
producer.close()  # 关闭生产者

# 关闭MySQL连接
cursor.close()  # 关闭游标
conn.close()  # 关闭数据库连接
