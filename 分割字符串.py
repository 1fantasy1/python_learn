str_1 = "stheima itcast boxuegu"
sum_1 = str_1.count("it")
str_2 = str_1.replace(" ","|")
list_1 = str_2.split("|")
print(f"字符串{str_1}中有：{sum_1}个it字符")
print(f"字符串{str_1}，被替换空格后，结果：{str_2}")
print(f"字符串{str_1}，按照|分割后，得到：{list_1}")
