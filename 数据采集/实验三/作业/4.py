from kafka import KafkaConsumer
import json

# Kafka配置
bootstrap_servers = ['localhost:9092']  # Kafka服务器地址
topic_name = 'test_topic'  # Kafka主题名称

# 创建Kafka消费者
# value_deserializer用于将接收到的字节数据转换为Python字典
consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# 订阅主题
consumer.subscribe([topic_name])

# 接收消息并计算总数
message_count = 0
try:
    for message in consumer:  # 循环接收消息
        message_count += 1  # 更新消息计数
        print(f"收到的消息: {message.value}")  # 打印接收到的消息
except KeyboardInterrupt:
    print(f"收到的消息总数: {message_count}")  # 打印消息总数
# 关闭消费者
consumer.close()  # 关闭消费者连接
