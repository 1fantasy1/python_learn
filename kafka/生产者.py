from kafka import KafkaProducer

# 连接 Kafka 集群
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# 准备要发送的消息，并转换为字节串类型
msg = "Hello World".encode('utf-8')

# 发送消息到名为 'test' 的主题
producer.send('test', msg)

# 关闭 KafkaProducer 实例，释放资源
producer.close()


'''from confluent_kafka import Producer

def delivery_report(err, msg):
    """处理消息发送的回调函数"""
    if err is not None:
        print("发送消息失败: {}".format(err))
    else:
        print("消息发送成功: {}".format(msg))

def produce_messages(bootstrap_servers, topic, messages):
    """生产者函数，用于发送消息到 Kafka 主题"""
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'client.id': 'kafka-producer'
    }

    # 创建生产者实例
    producer = Producer(**conf)

    # 发送消息
    for message in messages:
        producer.produce(topic, message.encode('utf-8'), callback=delivery_report)

    # 等待消息发送完成
    producer.flush()

    # 关闭生产者
    producer.close()

if __name__ == "__main__":
    bootstrap_servers = 'localhost:9092'  # Kafka 服务器地址
    topic = 'test'  # Kafka 主题
    messages = [
        "消息1",
        "消息2",
        "消息3"
    ]  # 要发送的消息列表

    # 发送消息到 Kafka 主题
    produce_messages(bootstrap_servers, topic, messages)
'''