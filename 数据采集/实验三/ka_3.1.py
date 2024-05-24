from kafka import KafkaProducer
import time
import uuid

# 创建Kafka生产者对象，指定Kafka服务器地址
producer = KafkaProducer(bootstrap_servers='localhost:9092')
display_interval = 5  # 设置显示消息生产速率的时间间隔（秒）

print('正在向主题 assign_topic 发送消息。按 Ctrl-C 终止。')
display_iteration = 0  # 记录显示消息生产速率的次数
message_count = 0  # 记录在一个时间间隔内生产的消息数量
start_time = time.time()  # 记录开始时间

try:
    while True:
        identifier = str(uuid.uuid4())  # 生成唯一标识符
        producer.send('assign_topic', identifier.encode('utf-8'))  # 发送消息到Kafka主题
        message_count += 1  # 增加消息数量计数
        now = time.time()  # 获取当前时间
        if now - start_time > display_interval:  # 如果超过了显示间隔时间
            print('第 %i 次迭代：%i 条消息在 %.0f 条消息/秒 的速率下生产' % (
                display_iteration,
                message_count,
                message_count / (now - start_time)))  # 打印消息生产速率
            display_iteration += 1  # 增加显示次数计数
            message_count = 0  # 重置消息数量计数
            start_time = time.time()  # 重置开始时间
except KeyboardInterrupt:
    print('消息生产已中断。')

finally:
    producer.close()  # 关闭Kafka生产者
    print('生产者已关闭。')