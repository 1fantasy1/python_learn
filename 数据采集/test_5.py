# 将学号为10003的学生从这三个表中删除。
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

try:
    # 删除选课表中与学号为 '10003' 的学生相关的记录
    sql_delete_sc = "DELETE FROM SC WHERE Sno = %s"
    data = ('10003',)
    cursor.execute(sql_delete_sc, data)

    # 删除学生表中学号为 '10003' 的学生记录
    sql_delete_student = "DELETE FROM Student WHERE Sno = %s"
    cursor.execute(sql_delete_student, data)

    # 提交事务
    connect.commit()

    print("成功删除学号为 '10003' 的学生及其选课信息。")

except Exception as e:
    # 发生异常时回滚
    connect.rollback()
    print("删除学生失败:", e)

finally:
    # 关闭数据库连接
    cursor.close()
    connect.close()