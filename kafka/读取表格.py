# import sys
# import json
# import pandas as pd
# import os
# from kafka import KafkaProducer
# from kafka import KafkaConsumer
# from kafka.errors import KafkaError
# import matplotlib.pyplot as plt
# import mysql.connector
#
# # 读取CSV文件
# data = pd.read_csv('D:\\测试\\score.csv')
#
# # MySQL数据库连接配置
# mysql_config = {
#     'host': 'localhost',        # MySQL服务器地址
#     'user': 'your_username',    # MySQL用户名
#     'password': 'your_password',# MySQL密码
#     'database': 'your_database_name' # MySQL数据库名称
# }
#
# # 连接到MySQL数据库
# conn = mysql.connector.connect(**mysql_config)
#
# # 创建MySQL数据库表
#  create_table_query = '''
# CREATE TABLE IF NOT EXISTS score_data (
#     ID INT AUTO_INCREMENT PRIMARY KEY,  # 主键ID，自增
#     Name VARCHAR(255),                  # 姓名字段
#     Score INT                           # 分数字段
# )
# '''
# cursor = conn.cursor()
# cursor.execute(create_table_query)
#
# # 将数据插入到MySQL数据库表
# insert_query = '''
# INSERT INTO score_data (Name, Score) VALUES (%s, %s)
# '''
# for _, row in data.iterrows():
#     cursor.execute(insert_query, (row['Name'], row['Score']))
#
# # 提交事务并关闭连接
# conn.commit()
# cursor.close()
# conn.close()
#
# print("数据已成功插入到MySQL数据库表中。")
#
# # Kafka服务器相关配置
# KAFKA_HOST = "localhost"  # Kafka服务器地址
# KAFKA_PORT = 9092         # Kafka服务器端口号
# KAFKA_TOPIC = "topic0"    # Kafka Topic名称
#
# # 读取CSV文件并转换为JSON格式
# data = pd.read_csv('D:\\测试\\score.csv')
# key_value = data.to_json()
#
# # Kafka生产者类
# class Kafka_producer():
#     def __init__(self, kafkahost, kafkaport, kafkatopic, key):
#         self.kafkaHost = kafkahost
#         self.kafkaPort = kafkaport
#         self.kafkatopic = kafkatopic
#         self.key = key
#         # 创建KafkaProducer实例
#         self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
#             kafka_host=self.kafkaHost,
#             kafka_port=self.kafkaPort)
#         )
#
#     # 发送JSON数据到Kafka
#     def sendjsondata(self, params):
#         try:
#             parmas_message = params
#             producer = self.producer
#             producer.send(self.kafkatopic, key=self.key, value=parmas_message.encode('utf-8'))
#             producer.flush()
#         except KafkaError as e:
#             print(e)
#
# # Kafka消费者类
# class Kafka_consumer():
#     def __init__(self, kafkahost, kafkaport, kafkatopic, groupid, key):
#         self.kafkaHost = kafkahost
#         self.kafkaPort = kafkaport
#         self.kafkatopic = kafkatopic
#         self.groupid = groupid
#         self.key = key
#         # 创建KafkaConsumer实例
#         self.consumer = KafkaConsumer(self.kafkatopic, group_id=self.groupid,
#                                       bootstrap_servers='{kafka_host}:{kafka_port}'.format(
#                                           kafka_host=self.kafkaHost,
#                                           kafka_port=self.kafkaPort)
#                                       )
#
#     # 消费Kafka中的数据
#     def consume_data(self):
#         try:
#             for message in self.consumer:
#                 yield message
#         except KeyboardInterrupt as e:
#             print(e)
#
# # 排序字典值函数
# def sortedDictValues(adict):
#     items = adict.items()
#     items = sorted(items, reverse=False)
#     return [value for key, value in items]
#
# # 主函数
# def main(xtype, group, key):
#     if xtype == "p":
#         # 生产模块
#         producer = Kafka_producer(KAFKA_HOST, KAFKA_PORT, KAFKA_TOPIC, key)
#         print("===========> producer:", producer)
#         params = key_value
#         producer.sendjsondata(params)
#     if xtype == 'c':
#         # 消费模块
#         consumer = Kafka_consumer(KAFKA_HOST, KAFKA_PORT, KAFKA_TOPIC, group, key)
#         print("===========> consumer:", consumer)
#         message = consumer.consume_data()
#         for msg in message:
#             msg = msg.value.decode('utf-8')
#             python_data = json.loads(msg)  # 将字符串转换成字典
#             key_list = list(python_data)
#             test_data = pd.DataFrame()
#             for index in key_list:
#                 if index == 'Name':
#                     a1 = python_data[index]
#                     data1 = sortedDictValues(a1)
#                     test_data[index] = data1
#                 else:
#                     a2 = python_data[index]
#                     data2 = sortedDictValues(a2)
#                     test_data[index] = data2
#             print(test_data)
#
#             # 绘制图表
#             # for column in test_data.columns:
#             #     plt.figure()
#             #     plt.bar(test_data.index, test_data[column])
#             #     plt.xlabel('Index')
#             #     plt.ylabel(column)
#             #     plt.title('Data for {}'.format(column))
#             #     plt.show()
#
# # 主程序入口
# if __name__ == '__main__':
#     main(xtype='p', group='py_test', key=None)  # 执行生产模块
#     main(xtype='c', group='py_test', key=None)  # 执行消费模块

import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
import pymysql
import csv

# MySQL数据库相关配置
MYSQL_HOST = "localhost"    # MySQL服务器地址
MYSQL_PORT = 3306           # MySQL服务器端口号
MYSQL_USER = "root"    # MySQL用户名
MYSQL_PASSWORD = "@Fantasy2024"  # MySQL密码
MYSQL_DB = "score"  # MySQL数据库名称
MYSQL_TABLE = "score"       # MySQL表名


# Kafka服务器相关配置
KAFKA_HOST = "localhost"    # Kafka服务器地址
KAFKA_PORT = 9092           # Kafka服务器端口号
KAFKA_TOPIC = "topic0"      # Kafka Topic名称

# 连接到 MySQL 数据库
connection = pymysql.connect(
    host = MYSQL_HOST,
    user = MYSQL_USER,
    password = MYSQL_PASSWORD,
    database = MYSQL_DB
)

# 创建一个游标对象
cursor = connection.cursor()

# 读取CSV文件并传入MySQL数据库
def csv_to_mysql():
    try:
        with open('D:\测试\score.csv', 'r') as file:
            csv_data = csv.reader(file)
            # 跳过标题行
            next(csv_data)
            # 遍历 CSV 数据并将其插入到数据库中
            for row in csv_data:
                cursor.execute('INSERT INTO score (Name, Score) VALUES (%s, %s)', row)
        # 提交更改并关闭连接
        connection.commit()
        connection.close()
    except Exception as e:
        print("Error:", e)

# Kafka生产者类
class Kafka_producer():
    def __init__(self, kafkahost, kafkaport, kafkatopic, key):
        self.kafkaHost = KAFKA_HOST
        self.kafkaPort = KAFKA_PORT
        self.kafkatopic = KAFKA_TOPIC
        self.key = key
        self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort)
        )

    # 从MySQL数据库读取数据并发送到Kafka
    def send_data_from_mysql(self):
        try:
            conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB)
            cursor = conn.cursor()
            # 从MySQL数据库读取数据
            cursor.execute("SELECT * FROM {}".format(MYSQL_TABLE))
            rows = cursor.fetchall()
            # 将数据转换为JSON格式并发送到Kafka
            for row in rows:
                data = dict(zip([desc[0] for desc in cursor.description], row))
                json_data = json.dumps(data)
                self.producer.send(self.kafkatopic, key=self.key, value=json_data.encode('utf-8'))
            self.producer.flush()
            print("数据成功发送到Kafka消息队列！")
        except KafkaError as e:
            print("Kafka Error:", e)
        except Exception as e:
            print("Error:", e)
        finally:
            cursor.close()
            conn.close()

# 主函数
def main():
    # 将CSV文件数据传入MySQL数据库
    csv_to_mysql()
    # 使用Kafka生产者将数据从MySQL数据库中读取并发送到Kafka消息队列
    producer = Kafka_producer(KAFKA_HOST, KAFKA_PORT, KAFKA_TOPIC, key=None)
    producer.send_data_from_mysql()

# 主程序入口
if __name__ == '__main__':
    main()
