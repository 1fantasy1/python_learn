from data_define import Record
from file_define import FileReader, TextFileReader, JsonFileReader

text_file_reader = TextFileReader("D:/2011年1月销售数据.txt")
json_file_reader = JsonFileReader("D:/2011年2月销售数据JSON.txt")

jan_data = text_file_reader.read_data()  # type:list[Record]
feb_data = json_file_reader.read_data()  # type:list[Record]

# 将2个月份的数据合并
all_data = jan_data + feb_data  # type:list[Record]

# 开始进行数据计算
data_dict = {}
for record in all_data:
    if record.date in data_dict.keys():
        #当前日期已经有记录了，所以和老日期做累加即可
        data_dict[record.date] += record.money
    else:
        # 没有记录日期，做新记录
        data_dict[record.date] = record.money

# 验证代码
print(data_dict)