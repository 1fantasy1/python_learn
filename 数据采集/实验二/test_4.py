# 将学号为10005的学生， OperatingSystems(00004)成绩为73分这一记录写入选课表中
import pymysql.cursors

# 连接数据库
connect = pymysql.Connect(
    host='localhost',    # 数据库主机地址
    port=3306,           # 数据库端口号
    user='root',         # 数据库用户名
    passwd='@Fantasy2024',  # 数据库密码
    db='school',         # 要连接的数据库名称
    charset='utf8'       # 字符集
)

# 获取游标
cursor = connect.cursor()

# 设置sql语句
sql = "INSERT INTO sc(Sno,Cno,Grade) VALUES('%s','%s',%d)"

# 设置数据
data = ('10005', '00004', 73)

# 执行sql语句，并获取执行结果
result = cursor.execute(sql % data)
connect.commit()

# 输出执行结果
print(result)

# 关闭数据库连接
cursor.close()
connect.close()