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

try:
    # 接收消息并计算每个消息的长度
    for message in consumer:  # 循环接收消息
        message_length = len(str(message.value))  # 计算消息长度
        print(f"收到的消息: {message.value} (长度为: {message_length})") # 打印消息和长度
except KeyboardInterrupt:
    # 关闭消费者
    consumer.close()  # 关闭消费者连接
