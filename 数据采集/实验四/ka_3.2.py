from kafka import KafkaConsumer, TopicPartition
import time

display_interval = 5  # 设置显示消息消费速率的时间间隔（秒）

# 创建Kafka消费者对象，指定Kafka服务器地址和偏移重置策略
consumer1 = KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest')
consumer1.assign([TopicPartition('assign_topic', 0)])  # 指定消费的主题和分区

print('正在消费主题 assign_topic 的消息。按 Ctrl-C 终止。')
display_iteration = 0  # 记录显示消息消费速率的次数
message_count = 0  # 记录在一个时间间隔内消费的消息数量
partitions = set()  # 记录消息来自的分区集合
start_time = time.time()  # 记录开始时间

try:
    while True:
        message = next(consumer1)  # 获取下一条消息
        identifier = str(message.value, encoding="utf-8")  # 解码消息内容
        message_count += 1  # 增加消息数量计数
        partitions.add(message.partition)  # 记录消息来自的分区
        now = time.time()  # 获取当前时间
        if now - start_time > display_interval:  # 如果超过了显示间隔时间
            print('第 %i 次迭代：%i 条消息在 %.0f 条消息/秒 的速率下消费 - 来自分区 %r' % (
                display_iteration,
                message_count,
                message_count / (now - start_time),
                sorted(partitions)))  # 打印消息消费速率和分区信息
            display_iteration += 1  # 增加显示次数计数
            message_count = 0  # 重置消息数量计数
            partitions = set()  # 重置分区集合
            start_time = time.time()  # 重置开始时间
except KeyboardInterrupt:
    print('消息消费已中断。')
finally:
    consumer1.close()  # 关闭Kafka消费者
    print('消费者已关闭。')
