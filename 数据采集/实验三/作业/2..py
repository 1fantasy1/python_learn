from kafka import KafkaProducer
import json

# Kafka配置
bootstrap_servers = ['localhost:9092']  # Kafka服务器地址
topic_name = 'test_topic'  # Kafka主题名称

# 创建Kafka生产者
# value_serializer用于将Python字典转换为JSON字符串
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# 发送消息
messages = [{'message': '你好!KAFKA.'}, {'message': '这是一条测试信息.'}]
for message in messages:
    producer.send(topic_name, value=message)  # 发送消息到Kafka主题
    print(f"发送消息: {message}")

# 关闭生产者
producer.flush()  # 确保所有消息都被发送
producer.close()  # 关闭生产者连接
