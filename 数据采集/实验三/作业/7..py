from kafka import KafkaConsumer
import json
import pymysql

# Kafka配置
bootstrap_servers = ['localhost:9092']  # Kafka服务器地址
topic_name = 'test_topic'  # Kafka主题名称

# MySQL配置
mysql_host = 'localhost'  # MySQL服务器地址
mysql_user = 'root'  # MySQL用户名
mysql_password = '@Fantasy2024'  # MySQL密码
mysql_db = 'test_db'  # MySQL数据库名
mysql_table = 'kafka_messages'  # MySQL表名

# 创建Kafka消费者
# value_deserializer用于将接收到的字节数据转换为Python字典
consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# 订阅主题
consumer.subscribe([topic_name])

# 连接到MySQL数据库
connection = pymysql.connect(host=mysql_host,
                             user=mysql_user,
                             password=mysql_password,
                             db=mysql_db,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # 接收消息并将其写入MySQL数据库
        for message in consumer:
            # 假设消息格式为 {'message': 'Hello, Kafka!'}
            message_text = message.value['message']
            # 将数据插入MySQL表
            sql = f"INSERT INTO `{mysql_table}` (`message_text`) VALUES (%s)"
            cursor.execute(sql, (message_text,))
            connection.commit()  # 提交事务
            print(f"收到的消息: {message_text} (已插入MySQL)")
except KeyboardInterrupt:
    consumer.close()  # 关闭消费者连接
    connection.close()  # 关闭数据库连接