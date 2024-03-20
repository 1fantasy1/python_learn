from data_define import Record
from file_define import FileReader, TextFileReader, JsonFileReader
from pymysql import Connection

text_file_reader = TextFileReader("D:/2011年1月销售数据.txt")
json_file_reader = JsonFileReader("D:/2011年2月销售数据JSON.txt")
list_1 = text_file_reader.read_data()
list_2 = json_file_reader.read_data()
all_data = list_1 + list_2

con = Connection(
    host='localhost',  # 主机名（IP）
    port=3306,  # 端口
    user='root',  # 账户名
    password='@Fantasy2024',  # 密码
    autocommit = True # 设置自动提交
)
# 获取游标对象
cursor = con.cursor()
# 选择数据库
con.select_db("py_sql")
# 执行sql语句
for record in all_data:
    sql = f"insert into orders(date , order_id ,money ,province)" \
          f" values('{record.date}','{record.order_id}','{record.money}','{record.province}')"
    cursor.execute(sql)

con.close()