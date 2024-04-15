import sys
import json
import pandas as pd
import os
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import matplotlib.pyplot as plt

# Kafka服务器相关配置
KAFKA_HOST = "localhost"  # Kafka服务器地址
KAFKA_PORT = 9092         # Kafka服务器端口号
KAFKA_TOPIC = "topic0"    # Kafka Topic名称

# 读取CSV文件并转换为JSON格式
data = pd.read_csv('D:\\测试\\score.csv')
key_value = data.to_json()

# Kafka生产者类
class Kafka_producer():
    def __init__(self, kafkahost, kafkaport, kafkatopic, key):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.key = key
        # 创建KafkaProducer实例
        self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort)
        )

    # 发送JSON数据到Kafka
    def sendjsondata(self, params):
        try:
            parmas_message = params
            producer = self.producer
            producer.send(self.kafkatopic, key=self.key, value=parmas_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)

# Kafka消费者类
class Kafka_consumer():
    def __init__(self, kafkahost, kafkaport, kafkatopic, groupid, key):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.groupid = groupid
        self.key = key
        # 创建KafkaConsumer实例
        self.consumer = KafkaConsumer(self.kafkatopic, group_id=self.groupid,
                                      bootstrap_servers='{kafka_host}:{kafka_port}'.format(
                                          kafka_host=self.kafkaHost,
                                          kafka_port=self.kafkaPort)
                                      )

    # 消费Kafka中的数据
    def consume_data(self):
        try:
            for message in self.consumer:
                yield message
        except KeyboardInterrupt as e:
            print(e)

# 排序字典值函数
def sortedDictValues(adict):
    items = adict.items()
    items = sorted(items, reverse=False)
    return [value for key, value in items]

# 主函数
def main(xtype, group, key):
    if xtype == "p":
        # 生产模块
        producer = Kafka_producer(KAFKA_HOST, KAFKA_PORT, KAFKA_TOPIC, key)
        print("===========> producer:", producer)
        params = key_value
        producer.sendjsondata(params)
    if xtype == 'c':
        # 消费模块
        consumer = Kafka_consumer(KAFKA_HOST, KAFKA_PORT, KAFKA_TOPIC, group, key)
        print("===========> consumer:", consumer)
        message = consumer.consume_data()
        for msg in message:
            msg = msg.value.decode('utf-8')
            python_data = json.loads(msg)  # 将字符串转换成字典
            key_list = list(python_data)
            test_data = pd.DataFrame()
            for index in key_list:
                if index == 'Name':
                    a1 = python_data[index]
                    data1 = sortedDictValues(a1)
                    test_data[index] = data1
                else:
                    a2 = python_data[index]
                    data2 = sortedDictValues(a2)
                    test_data[index] = data2
            print(test_data)

            # 绘制图表
            # for column in test_data.columns:
            #     plt.figure()
            #     plt.bar(test_data.index, test_data[column])
            #     plt.xlabel('Index')
            #     plt.ylabel(column)
            #     plt.title('Data for {}'.format(column))
            #     plt.show()

# 主程序入口
if __name__ == '__main__':
    main(xtype='p', group='py_test', key=None)  # 执行生产模块
    main(xtype='c', group='py_test', key=None)  # 执行消费模块