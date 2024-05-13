# 查询学号为10002学生的所有成绩，结果中需包含学号、姓名、所在系别、课程号、课程名以及对应成绩。
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
sql = "SELECT student.Sno,Sname,Sdept,course.Cno,Cname,Grade "\
      "FROM student,course,sc "\
      "WHERE student.Sno = sc.Sno AND course.Cno = sc.Cno AND sc.Sno = '%s'"

# 设置数据
data = ('10002',)

# 执行sql语句
cursor.execute(sql % data)

# 获取数据
print("一共有%s条记录" % cursor.rowcount)  # 打印查询结果的行数
for row in cursor.fetchall():  # 遍历查询结果集
    print("学号：%s\t姓名：%s\t系别：%s\t课程号：%s\t课程名：%s\t成绩：%d" % row)  # 打印每行结果

# 关闭数据库连接
cursor.close()
connect.close()