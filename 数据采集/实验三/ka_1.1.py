import json
from kafka import KafkaProducer
from pymysql import Connection

# 创建Kafka生产者对象，指定Kafka服务器地址
producer = KafkaProducer(bootstrap_servers='localhost:9092',value_serializer=lambda v:json.dumps(v).encode('utf-8'))

# 创建MySQL数据库连接对象，指定连接参数
con = Connection(
    host='localhost',  # 主机名（IP）
    port=3306,  # 端口
    user='root',  # 账户名
    password='@Fantasy2024'  # 密码
)

# 创建游标对象，用于执行SQL语句
cursor = con.cursor()

# 定义SQL查询语句，查询学生表中的学号、姓名、性别、年龄
sql = "select sno,sname,ssex,sage from student.student;"
cursor.execute(sql)  # 执行SQL查询
data = cursor.fetchall()  # 获取查询结果
con.commit()  # 提交事务

# 遍历查询结果，将每条记录发送到Kafka主题
for msg in data:
    res = {}
    res['sno'] = msg[0]  # 学号
    res['name'] = msg[1]  # 姓名
    res['sex'] = msg[2]  # 性别
    res['age'] = msg[3]  # 年龄
    producer.send('mysql_topic', res)  # 发送消息到Kafka主题

# 关闭数据库连接
con.close()
# 关闭Kafka生产者
producer.close()
