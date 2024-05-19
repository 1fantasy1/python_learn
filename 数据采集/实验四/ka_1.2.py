from kafka import KafkaConsumer
import json

# 创建Kafka消费者对象，指定要消费的主题、Kafka服务器地址、消费者组ID和偏移重置策略
consumer = KafkaConsumer(
    'mysql_topic',  # 要消费的Kafka主题
    bootstrap_servers=['localhost:9092'],  # Kafka服务器地址
    group_id=None,  # 消费者组ID，设为None表示独立消费
    auto_offset_reset='earliest'  # 偏移重置策略，'earliest'表示从最早的消息开始消费
)

# 循环消费Kafka中的消息
print('正在消费主题 mysql_topic 的消息。按 Ctrl-C 终止。')

try:
    for msg in consumer:
        msg1 = str(msg.value, encoding="utf-8")  # 将消息的字节数据转换为字符串
        data = json.loads(msg1)  # 将JSON字符串转换为Python字典
        print(data)  # 打印消息数据
except KeyboardInterrupt:
    print('消息消费已中断。')
finally:
    consumer.close()  # 关闭Kafka消费者
    print('消费者已关闭。')
