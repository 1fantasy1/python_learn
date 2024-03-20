from pymysql import Connection

con = Connection(
    host='localhost',  # 主机名（IP）
    port=3306,  # 端口
    user='root',  # 账户名
    password='@Fantasy2024',  # 密码
    # autocommit = True # 设置自动提交
)
# print(con.get_server_info())
# 获取游标对象
cursor = con.cursor()
# 选择数据库
con.select_db("world")
# 执行sql语句
cursor.execute("insert into student values(10022 ,'李轩辕' ,29 ,'男')")
# 通过commit确认
# con.commit()
# 关闭链接
con.close()