from pymysql import Connection

con = Connection(
    host='localhost',  # 主机名（IP）
    port=3306,  # 端口
    user='root',  # 账户名
    password='@Fantasy2024'  # 密码
)

# print(con.get_server_info())
# 获取游标对象
cursor = con.cursor()
# 选择数据库
con.select_db("world")
# execute 执行Mysql语句
# cursor.execute("create table test_pymysql (id int)") # 代码中分号可不写

# cursor.execute("select * from student")
# # fetchall 执行查询返回结果 , 封装到元组中
# results = cursor.fetchall()
# # print(results)
# for r in results:
#     print(r)

# 关闭链接
# con.close()
