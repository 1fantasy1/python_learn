from kafka import KafkaProducer
import json

# 打开一个json文件
with open("./data.json") as data:
    # 将json文件内容转换为python对象
    strJson = json.load(data)

# 创建Kafka生产者对象，并指定value_serializer参数用于将数据序列化为JSON格式
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Kafka服务器地址
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # 序列化函数，将数据转换为JSON字符串并编码为UTF-8
)

# 发送消息到指定的Kafka主题
producer.send('json_topic', strJson)

# 关闭Kafka生产者
producer.close()