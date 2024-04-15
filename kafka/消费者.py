from kafka import KafkaConsumer

# 创建 Kafka 消费者实例
consumer = KafkaConsumer(
    # 指定要订阅的主题（可以是一个字符串或字符串列表）
    'test',
    # 指定 Kafka 集群的启动服务器地址和端口号
    bootstrap_servers=['localhost:9092'],
    # 指定消费者组的 ID，设置为 None 表示不加入任何消费者组
    group_id=None,
    # 指定当没有初始偏移量或偏移量无效时消费者该如何处理
    auto_offset_reset='smallest',  # 从最早的消息开始消费，可选的值还有 'largest' 表示从最新的消息开始消费
)

for msg in consumer:
    # 从消息对象中提取相关信息
    recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
    # 打印接收到的消息信息
    print(recv)

'''
from confluent_kafka import Consumer, KafkaError

def consume_messages(bootstrap_servers, group_id, topic):
    """消费者函数，用于从 Kafka 主题中接收消息"""
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }

    # 创建消费者实例
    consumer = Consumer(**conf)

    # 订阅主题
    consumer.subscribe([topic])

    try:
        while True:
            # 从 Kafka 主题中拉取消息
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # 当分区结束时，继续下一个消息
                    continue
                else:
                    # 其他错误
                    print("消费消息时发生错误: {}".format(msg.error()))
                    break

            # 打印消息
            print("接收到消息: {}".format(msg.value().decode('utf-8')))

    finally:
        # 关闭消费者
        consumer.close()

if __name__ == "__main__":
    bootstrap_servers = 'localhost:9092'  # Kafka 服务器地址
    group_id = 'my_consumer_group'  # 消费者组ID
    topic = 'your_topic'  # Kafka 主题

    # 接收消息并打印
    consume_messages(bootstrap_servers, group_id, topic)
'''
