# 由于培养计划改，现需将课程号为00001、课程名为DataBase的学分改为5学分
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
sql = "UPDATE course SET Credit = %d " \
      "WHERE Cno = '%s'"

# 设置数据
data = (5, '00001')

# 执行sql语句，并获取执行结果
result = cursor.execute(sql % data)

# 提交事务
connect.commit()

# 查看执行结果
print(result)

# 关闭数据库连接
cursor.close()
connect.close()
