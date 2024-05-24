from kafka import KafkaConsumer
import json

# Kafka配置
bootstrap_servers = ['localhost:9092']  # Kafka服务器地址
topic_name = 'test_topic'  # Kafka主题名称
output_file = '输出.txt'  # 输出文件名

# 创建Kafka消费者
# value_deserializer用于将接收到的字节数据转换为Python字典
consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# 订阅主题
consumer.subscribe([topic_name])

try:
    # 接收消息并将其保存到文件
    with open(output_file, 'w') as f:  # 打开文件准备写入
        for message in consumer:  # 循环接收消息
            print(f"收到的消息: {message.value}")  # 打印接收到的消息
            f.write(f"{message.value}\n")  # 将消息写入文件
except KeyboardInterrupt:
    print(f"已将消息保存，文件名为{output_file}")  # 打印消息总数
finally:
    # 关闭消费者
    consumer.close()  # 关闭消费者连接
