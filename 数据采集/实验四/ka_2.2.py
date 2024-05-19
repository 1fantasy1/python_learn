from kafka import KafkaConsumer
import json

class Consumer:
    def __init__(self):
        self.server = 'localhost:9092'
        self.topic = 'json_topic'
        self.consumer = None
        self.consumer_timeout_ms = 5000  # 消费者超时时间，单位为毫秒
        self.group_id = 'test1'  # 消费者组ID

    def get_connect(self):
        # 创建Kafka消费者对象
        self.consumer = KafkaConsumer(
            self.topic,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            bootstrap_servers=self.server,
            enable_auto_commit=False,  # 禁用自动提交偏移量
            consumer_timeout_ms=self.consumer_timeout_ms
        )

    def begin_consumer(self):
        try:
            while True:
                for message in self.consumer:
                    data = message.value.decode('utf-8')  # 解码消息内容
                    data = json.loads(data)  # 将JSON字符串转换为Python字典
                    print(data)  # 打印消息数据
                    # 提交当前消息的偏移量
                    self.consumer.commit()
        except KeyboardInterrupt:
            print("消费已中止")
        finally:
            self.consumer.close()  # 关闭消费者

# 创建Consumer对象并启动消费
c = Consumer()
c.get_connect()
c.begin_consumer()