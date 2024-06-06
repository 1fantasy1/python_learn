# 钱包余额
# money = 50
# 购买冰淇淋10元，可乐5元
# money = money -10 -5
# print("当前还剩：",money)
# name = "小明"
# age = 18
# %s是字符串，%d是整数，%f是浮点数,可在字母前加入数字控制
# 如：%7.2d = --00.00之类，前面数字指总位数，包括小数点，--表示空格，小数区域会四舍五入
# print("我是：" + name + "，我的年龄是：%s" % age)
# print(f"我是：{name}，我的年龄是：{age}")#不关心类型，不做精度控制
# name = "沧海传播"
# stock_price = 6.66
# stock_code = 114514
# stock_price_growth_factor = 6.66
# growth_days = 6
# print(f"公司：{name}，股票代码：{stock_code}，当前股价：{stock_price}")
# print("每日增长系数是：%.2f，经过%d天的增长后，股价达到了：%.2s" % (stock_price_growth_factor,growth_days,growth_days*stock_price_growth_factor))
# a = input("a是多少")#input默认输出字符串！！！
# print("a是%s" % a)
# user_name = input("你的用户名称")
# user_type = input("你的用户类型")
# print("您好%s,您是珍贵的：%s用户，欢迎您的光临。" % (user_name, user_type))
# bool_1 = True
# bool_2 = False
# print(f"bool_1变量的内容是：{bool_1}，类型是：{type(bool_1)}")
# print(f"bool_2变量的内容是：{bool_2}，类型是：{type(bool_2)}")
# print("欢迎来到游乐园，儿童免费，成人收费。")
# age = int(input("请输入你的年龄："))
# 最优化理论 age >= 18:#冒号不能忘
#     print("您已成年，游玩需要补票10元。")
# print("祝您游玩愉快。")
# print("欢迎来到沧海动物园。")
# text_height = int(input("请输入你的身高(cm)："))
# 最优化理论 text_height > 120:
#     print("您的身高超出120cm，游玩需要补票10元。")
# else:
#     print("您的身高未超出120cm，可以免费游玩。")
# print("祝您游玩愉快。")
# num = int(input(""))
# 最优化理论 int(input("请输入第一次猜想的数字：")) == num:
#     print("猜对了")
# elif int(input("不对，再猜一次：")) == num:
#     print("猜对了")
# elif int(input("不对，再猜最后一次：")) == num:
#     print("猜对了")
# else:
#     print("Sorry，全部猜错啦，我想的是：%d" % num)
# name = "沧海有泪"
# for i in name:
#     #将name中的元素逐个取出
#     print(i)
# for i in range(10):
#     print(i,end=" ")
# print("")
# for i in range(5,10):
#     print(i,end=" ")
# print("")
# for i in range(5,15,2):
#     print(i,end=" ")
# for i in range(1,5):
#     print("语句一")
#     for i in range(1,5):
#         print("语句二",end=" ")
#         continue#直接结束当前循环
#         print("语句三")
#     print("")
#     print("语句四")
# for i in range(1,100):
#     print("语句一")
#     break# 直接终止循环
#     print("语句二")
# print("语句三")
# def say_hello():
#     print("Hello")
# say_hello()
# None等同于False
# None可用于声明无初始值的变量，例：a = None
# num = 100
# def texta():
#     print(num)
# def textb():
#     global num # global 可声明num是全局变量
#     num = 200
#     print(num)
# texta()
# textb()
# print(num)

# #列表用中括号[]定义
# name_list = ["Tom", "Lisy", "Rose"]
# print(name_list[0])#下标索引，正向从0开始
# print(name_list[1])
# print(name_list[2])
# print(name_list[-3])#下标索引，反向从-1开始
# print(name_list[-2])
# print(name_list[-1])

# name_list = [["Tom", "Lisy"], ["Rose", "Boy"]]
# print(name_list)
# print(name_list[0][0])
# print(name_list[0][1])
# print(name_list[1][0])
# print(name_list[1][1])#列表中的列表需要分别写，反向也一样。
# # print(name_list[1][2])#且不能超出列表范围！！！会报错。
#
# i = name_list.index(["Tom","Lisy"])
# #index用于查找元素下标，正向！
# #语法为    列表.index(元素)
# print(i)
# # o = name_list.index("Tom")#未找到会报错，错误为‘ValueError’
#
# name_list[0] = "Mike"#修改对应下标的元素
# print(name_list)
#
# name_list.insert(1, "Jerry")
# #insert用于插入元素。
# #语法为    列表.insert(a,b)
# #其中a为插入元素后的下标，b为要插入元素。
# print(name_list)
#
# #追加元素方法一
# name_list.append("Mary")
# #append用于追加元素到列表末尾
# #语法为    列表.append(元素)
# print(name_list)
#
# #追加元素方法二
# list2 = ["John","Linda","Jessica"]
# name_list.extend(list2)
# #extend用于追加其他数据容器到列尾。
# #方法是将其他数据容器内容取出，再依次添加到列尾
# #语法为    列表.extend(其他数据容器)
# print(name_list)
#
# #删除元素方法一
# del name_list[-1]
# #del用于删除对应下标的元素。
# #语法为    del 列表[下标]
# print(name_list)
#
# #删除元素方法二
# g = name_list.pop(-1)
# #pop用于删除对应下标的元素。
# #语法为    列表.pop(下标)
# #pop还可将删除的元素作为返回值输出
# print(name_list)
# print(g)
#
# name_list.insert(3,["Rose","Boy"])
# print(name_list)
#
# name_list.remove(["Rose","Boy"])
# #remove用于某元素在列表中的第一个匹配项。
# #语法为    列表.remove(元素)
# print(name_list)
#
# name_list.clear()
# #clear用于删除整个列表。
# #语法为    列表.clear()
# print(name_list)
#
# name_list = [1,1,1,2,2,3]
# print(name_list)
#
# v = name_list.count(1)
# #count用于统计指定元素的个数。
# #语法为    列表.count()
# print(v)
#
# k = len(name_list)
# #len用于统计列表中总共有多少元素
# #语法为    len(列表)
# print(k)
# def list_while_func():
#     my_list = ["Tom", "Lisy", "Rose", "Boy"]
#     q = 0
#     while q < len(my_list):
#         element = my_list[q]
#         print(f"列表的元素是{element}")
#         q += 1
# list_while_func()

# def list_for_func():
#     my_list = ["Tom", "Lisy", "Rose", "Boy"]
#     for dan in my_list:
#         print(f"列表中的元素是：{dan}")
# list_for_func()
# 元组用小括号()定义
# yuan_1 = (1,2,3,4,5,6,7,8,9)
# yuan_2 = (1, )#元组只有一个元素后面要加逗号.
# yuan_3 = ((1,2),(3,4))
# print(f"{type(yuan_1)},{type(yuan_2)},{type(yuan_3)}")
# num = yuan_3[1][1]
# print(num)
# """
# !!!
# 元组只有index(),count(),len(元组)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
# !!!
# """
#
# #元组不可修改(如增加或删除元素),但有特殊情况,元组内有可修改的数据容器就可修改
# yuan_4 = (1,2,[3,4])
# yuan_4[2][1] = 5
# print(yuan_4)
#字符串无法修改,修改会报错.
# str_1 = "3.1415926535"
# # 取任意下标的元素.
# num = str_1[2]
# print(f"在字符串{str_1}中下标2的元素为：{num}")
#
# # 字符串.index(元素) 方法 查询下标
# num_2 = str_1.index("1")
# print(f"在字符串{str_1}中查找1，其起始下标是：{num_2}")
#
# # 字符串.replace(a.b) 方法 字符串替换
# # a为替换前的字符串，b需要替换的字符串。
# str_2 = "114514"
# str_3 = str_2.replace("11","22")
# print(f"替换之后的字符串为{str_3}")
#
# # 字符串.split(分隔符字符串) 方法 以分隔符切分为列表。
# str_4 = "Tom Lisy Rose Boy Mike"
# list_1 = str_4.split(" ")
# print(f"将字符串{str_4}进行split切分后得到：{list_1}，类型是：{type(list_1)}")
#
# # 字符串.strip() 方法 strip可传参。
# str_5 = "    fantasy    "
# str_6 = "1122fantasy2211"
# print(f"字符串{str_5}被sprit()后变为{str_5.strip()}")
# # 不输入参数为：取出字符串前后空格，以及换行符。
# print(f"字符串{str_6}被sprit(12)后变为{str_6.strip('12')}") #单引号
# # 输入参数为：去除指定前后字符串，是按照单个字符去除
# # 如传入"12"是："1"，"2"都会移除。
#
# #字符串.count() 方法 统计字符串某字符出现次数。
# str_7 = "3.1415926535"
# sum = str_7.count("1")
# print(f"字符串{str_7}中1出现的次数为{sum}")
#
# # len(字符串) 方法 统计所有字符个数
# num_3 =len(str_1)
# print(f"在字符串{str_1}元素总数为{num_3}")

# 序列必须内容连续，有序，支持下标索引。
# 对序列进行切片操作
# 格式为 新的数据容器 = 要切片的数据容器[a:b:c]
# 其中a为起始下标，b为结束下标，c为步长。
# 取得切片不包括b本身，c等于1可不写，c可为负数。
# 取到两端可不写
# list_1 = [1,2,3,4,5,6,7,8,9,10]
# list_2 = list_1[2:5] #步长为1可不写！
# print(list_2)
#
# yuan_1 = (11,12,13,14,15,16,17,18,19,20)
# yuan_2 = yuan_1[::2] # 取到两端可不写！
# print(yuan_2)
#
# str_1 = "3.1415926535"
# str_2 = str_1[-1:-8:-2] # 步长为负数时，起始结束也要为负！
# print(str_2)
# 集合会自动去重
# 集合不支持下标索引
# 元素进入会打乱顺序
# set_0 = set()
# set_1 = {"Mike","John","Tom","Tom","John","John"}
# print(set_1,type(set_1))
#
# # add 方法
# set_1.add("Jerry")
# print(f"添加Jerry后为{set_1}")
#
# # remove 方法
# set_1.remove("John")
# print(f"移除John后的集合为{set_1}")
#
# # pop 方法
# # 没有参数，随机取出！！！取出后原集合的元素被删除，输出到新集合中。
# set_2 = set_1.pop()
# print(f"从集合set_1中随机取出的元素是{set_2}")
# print(f"集合set_1被pop方法随机取出元素后为：{set_1}")
#
# # clear 方法
# set_1.clear()
# print(f"集合被清空后为：{set_1}") # set()为空集
#
# # difference 方法
# # 格式为：集合一.difference(集合二)
# # 取出两集合的差集，得到一个新集合，原集合不变。
# # 取出的是集合一有而集合二没有的元素。
# set_3 = {1,2,3,4,5,6,7,8,9}
# set_4 = {2,3,4,5,6,10,11}
# set_5 = set_3.difference(set_4)
# print(f"集合{set_3}与集合{set_4}的差集为{set_5}")
#
# # difference_update 方法
# # 格式为 集合一.difference_update(集合二)
# # 在集合一中，删除和集合二相同的元素。
# # 得到一个新集合。
# # 集合一被修改，集合二不变。
# set_6 = {1,2,3,7,8,9}
# set_7 = {2,3,4,5,6,7,10,11}
# set_8 = set_3.difference_update(set_4)
# print(f"集合{set_6}删除与集合{set_7}相同的元素后为：{set_8}") # None为空集
#
# # union 方法
# # 合并集合
# # 格式为 集合一.union(集合二)
# # 得到一个新集合，集合一二不变。
# set_9 = {1,2,3}
# set_10 = {1,4,5}
# set_11 = set_9.union(set_10)
# print(f"{set_9}与{set_10}组成的新集合为：{set_11}")
#
# # len 方法
# # 统计集合元素个数
# # 格式为 sum = len(集合)
# set_12 = {1,1,4,5,1,4}
# sum_1 = len(set_12)
# print(f"集合{set_12}的元素个数为：{sum_1}个")
#
# # 集合的遍历
# # 集合不能通过while循环遍历，因为没有下标
# # 但可以用for循环遍历
# set_13 = {3,1,4,1,5,9,2,6,5,3,5}
# for i in set_13:
#     print(f"集合的元素有：{i}")
#     i += 1

# 字典的格式为{"key":Value,"key":Value,"key":Value}
# 字典的key不能重复，且不能为字典。
# Value 可为任意数据类型，表明字典可以嵌套。
# dict_0 = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5}
# dict_1 = {}
# dict_2 = dict()
# dict_3 = {
#     "Tom":{
#         "语文":99,
#         "数学":88,
#         "英语":77
#     },
#     "Jerry":{
#         "语文":66,
#         "数学":55,
#         "英语":44
#     },
#     "Mike":{
#         "语文":33,
#         "数学":22,
#         "英语":11
#     }
# }
# print(dict_0)
# print(dict_1)
# print(dict_2)
# print(f"学生的成绩是：{dict_3}")
#
# # 取对应key的Value值。
# # 看一下Tom的语文成绩
# num = dict_0["c"]
# print(num)
# language = dict_3["Tom"]["语文"]
# print(f"Tom的语文成绩是：{language}")
#
# # 字典新增，更新。
# # 语法： 字典[key] = Value
# # 结果： 字典被修改，新增了元素。如输入已有key，则更新元素。
# dict_0["f"] = "6"
# print(f"字典新增元素后为：{dict_0}")
# dict_0["f"] = "7"
# print(f"字典更新元素后为：{dict_0}")
#
# # 字典元素删除
# # 语法： 字典.pop(key)
# # 结果： 字典 = {}
# # 删除对应key的元素，并返回值。
# del_0 = dict_0.pop("f")
# print(f"删除元素{del_0}后的字典为：{dict_0}")
#
# # 清空字典元素
# # 语法： 字典.clear()
# dict_0.clear()
# print(f"清空字典元素后的字典为：{dict_0}")
#
# # 获取字典的全部key
# # 语法： 字典.keys()
# # 结果： 得到字典全部key
# key_1 = dict_3.keys()
# print(f"字典的全部key为：{key_1}")
#
# # 遍历字典
# # 字典没有下标，不能通过while循环遍历F！！！
# # 方式一：通过for循环遍历
# for key in dict_3.keys():
#     print(f"字典key是{key}。")
#     print(f"key:{key}对应的Value是{dict_3[key]}。")
#
# # 方式二：直接对字典进行for循环，每一次循环都直接得到key。
# for key in dict_3:
#     print(f"2字典key是{key}。")
#     print(f"key:{key}对应的Value是{dict_3[key]}。")
#
# # 统计字典的元素数量
# # 方法： len(字典)
# # 输出数量
# sum_1 = len(dict_3)
# print(f"字典{dict_3}中元素的个数为：{sum_1}")

# 容器的通用操作
# 如 len(), max(), min(),list(),tulpe(),str(),set()

# # 定义数据容器
# my_list = [1,2,3,4,5]
# my_tuple = (1,2,3,4,5)
# my_str = "hello world"
# my_set = {1,2,3,4,5}
# my_dict = {"key1":1, "key2":2, "key3":3,"key4":4,"key5":5}
#
# # len 元素个数
# print(f"列表 元素个数有：{len(my_list)}")
# print(f"元组 元素个数有：{len(my_tuple)}")
# print(f"字符串 元素个数有：{len(my_str)}")
# print(f"集合 元素个数有：{len(my_set)}")
# print(f"字典 元素个数有：{len(my_dict)}")
# print()
#
# # max 最大元素
# print(f"列表 最大元素是：{max(my_list)}")
# print(f"元组 最大元素是：{max(my_tuple)}")
# print(f"字符串 最大元素是：{max(my_str)}")
# print(f"集合 最大元素是：{max(my_set)}")
# print(f"字典 最大元素是：{max(my_dict)}")
# print()
#
# # min 最小元素
# print(f"列表 最小元素是：{min(my_list)}")
# print(f"元组 最小元素是：{min(my_tuple)}")
# print(f"字符串 最小元素是：{min(my_str)}")
# print(f"集合 最小元素是：{min(my_set)}")
# print(f"字典 最小元素是：{min(my_dict)}")
# print()
#
# # list(容器) 转为列表
# print(f"列表转列表的结果是：{list(my_list)}")
# print(f"元组转列表的结果是：{list(my_tuple)}")
# print(f"字符串转列表的结果是：{list(my_str)}")
# print(f"集合转列表的结果是：{list(my_set)}")
# print(f"字典转列表的结果是：{list(my_dict)}")
# print()
#
# # tulpe(容器) 转为元组
# print(f"列表转元组的结果是：{tuple(my_list)}")
# print(f"元组转元组的结果是：{tuple(my_tuple)}")
# print(f"字符串转元组的结果是：{tuple(my_str)}")
# print(f"集合转元组的结果是：{tuple(my_set)}")
# print(f"字典转元组的结果是：{tuple(my_dict)}")
# print()
#
# # str(容器) 转为字符串
# print(f"列表转字符串的结果是：{str(my_list)}")
# print(f"元组转字符串的结果是：{str(my_tuple)}")
# print(f"字符串转字符串的结果是：{str(my_str)}")
# print(f"集合转字符串的结果是：{str(my_set)}")
# print(f"字典转字符串的结果是：{str(my_dict)}")
# print()
#
# # set(容器) 转为集合
# print(f"列表转集合的结果是：{set(my_list)}")
# print(f"元组转集合的结果是：{set(my_tuple)}")
# print(f"字符串转集合的结果是：{set(my_str)}")
# print(f"集合转集合的结果是：{set(my_set)}")
# print(f"字典转集合的结果是：{set(my_dict)}")
#
# # 重新定义数据容器，为下面做准备。
# my_list = [3,2,1,5,4]
# my_tuple = (3,2,1,5,4)
# my_str = ("hello world")
# my_set = {3,2,1,5,4}
# my_dict = {"key3":1, "key2":2, "key1":3,"key5":4,"key4":5}
#
# # sorted(容器) 通用正向排序功能
# # 排完序后变为列表对象！！！
#
# print(f"列表对象排序的结果是：{sorted(my_list)}")
# print(f"元组对象排序的结果是：{sorted(my_tuple)}")
# print(f"字符串对象排序的结果是：{sorted(my_str)}")
# print(f"集合对象排序的结果是：{sorted(my_set)}")
# print(f"字典对象排序的结果是：{sorted(my_dict)}")
#
# # sorted(容器,[reverse = True]) 通用反向排序功能
# # 排完序后变为列表对象！！！
#
# print(f"列表对象反向排序的结果是：{sorted(my_list,reverse = True)}")
# print(f"元组对象反向排序的结果是：{sorted(my_tuple,reverse = True)}")
# print(f"字符串对象反向排序的结果是：{sorted(my_str,reverse = True)}")
# print(f"集合对象反向排序的结果是：{sorted(my_set,reverse = True)}")
# print(f"字典对象反向排序的结果是：{sorted(my_dict,reverse = True)}")

# 文件打开
# f = open("D:\测试.txt","r",encoding="utf-8")
# print(type(f))
# sum_1 = f.read(3)
# print(sum_1)
# # time.sleep(50000)
# f.close()

# import time
# time.sleep(1)
# from time import sleep
# sleep(2)
# time.sleep(2)
# from time import *
# sleep(5)
# from time import sleep as shui
# shui(5)
# 导入自定义包的模块，并使用。
# 方法一
# import learn.module_1
# import learn.module_2
# learn.module_1.info_print1()
# learn.module_2.info_print2()

# 方法二
# from learn.module_1 import info_print1
# from learn.module_2 import info_print2
# learn.module_1.info_print1()
# learn.module_2.info_print2()

# 方法三
# from learn.module_1 import info_print1
# from learn.module_2 import info_print2
# info_print1()
# info_print2()

# 有__all__的情况下：如；__all__ = ["module_1"]
# from learn import *
# learn.module_1.info_print1()
# # learn.module_2.info_print2() # 会报错！！！

# import my_utils.str_util
# from my_utils import file_util
# print(my_utils.str_util.str_reverse('hello'))
# print(my_utils.str_util.substr("hello", 1, 3))
#
# file_util.append_to_file("D:/hello.txt","hello")
# file_util.print_file_info("D:/hello.txt")

# # 设计一个类
# class Student:
#     name = None         # 记录学生姓名
#     gender = None       # 记录学生性别
#     nationality = None  # 记录学生国籍
#     native_place = None # 记录学生籍贯
#     age = None          # 记录学生年龄
#
# # 创建一个对象
# stu_1 = Student()
# stu_2 = Student()
#
# # 对对象属性进行赋值
# stu_1.name = "小明"
# stu_1.gender = "男"
# stu_1.nationality = "中国"
# stu_1.native_place = "北京"
# stu_1.age = 18
#
# # 输出对象属性
# print(stu_1.name)
# print(stu_1.gender)
# print(stu_1.nationality)
# print(stu_1.native_place)
# print(stu_1.age)

# # 定义一个带成员方法的类
# class Student():
#     name = None # 学生姓名
#
#     # self 必须写
#     def say_hi(self):
#         print(f'大家好，我是{self.name},大家多多关照')
#
#     def say_hi2(self,msg):
#         print(f"大家好，我是{self.name},{msg}")
#
# stu = Student()
# stu.name = "小明"
# stu.say_hi()
#
# stu2 = Student()
# stu2.name = "小王"
# stu2.say_hi()
#
# stu = Student()
# stu.name = "小明"
# stu.say_hi2("哎呦，不错哦")
#
# stu2 = Student()
# stu2.name = "小王"
# stu2.say_hi2("小伙子，我看好你哟")

# class Clock:
#     id = None # 序列号
#     price = None # 价格
#
#     def ring(self):
#         import winsound
#         winsound.Beep(500,500)
#
# # 构建两个闹钟对象并使其工作
# # 闹钟一
# clock_1 = Clock()
# clock_1.id = "0001"
# clock_1.price = 19.99
# print(f"闹钟的序列号是{clock_1.id}，价格是{clock_1.price}")
# clock_1.ring()
#
# #闹钟二
# clock_2 = Clock()
# clock_2.id = "0002"
# clock_2.price = 29.99
# print(f"闹钟的序列号是{clock_2.id}，价格是{clock_2.price}")
# clock_2.ring()

# class Student():
#     name = None
#     age = None
#     tel = None
#
#     # __init__ 会自动执行
#     # self不能忘！！！
#     def __init__(self, name, age, tel):
#         self.name = name
#         self.age = age
#         self.tel = tel
#     print("Student类创建了一个类对象") # 这条语句已经执行了
#     def __str__(self):
#         # __str__字符串方法
#         # 当类对象需要被转化为字符串时，会输出对应结果
#         return f"Student类对象， name = {self.name}, age = {self.age}, tel = {self.tel}"
#     def __lt__(self, other): # __lt__小于符号比较
#         return self.age < other.age
#
#     def __le__(self, other): # __le__小于等于比较
#         return self.age <= other.age
#
#     def __eq__(self, other):
#         # __eq__ 比较运算符，比较是否相等
#         # 如果不加，==或!=则会比较内存地址。
#         return self.age == other.age
#
# stu_1 = Student("小明", "18", "123456789")
# stu_2 = Student("小李","20","987654321")
#
# print(stu_1.name)
# print(stu_1.age)
# print(stu_1.tel)
# print(stu_2.name)
# print(stu_2.age)
# print(stu_2.tel)
#
# #字符串方法输出结果
# print(stu_1)
# print(str(stu_2))
#
# # 进行小于符号比较年龄
# print(stu_1 < stu_2)
# print(stu_1 > stu_2)
#
# # 进行小于等于符号比较年龄
# print(stu_1 <= stu_2)
# print(stu_1 >= stu_2)
#
# # 比较运算符判断年龄是否相等
# print(stu_1 == stu_2)
# print(stu_1 != stu_2)
# class Phone:
#     IMEI = 866224065939984
#     producer = "Apple"
#     __current_voltage = 2 # 当前手机电压
#
#     def __keep_single_core(self):
#         print("让CPU以单核模式运行")
#     def call_by_4g(self):
#         最优化理论 self.__current_voltage >= 1:
#             print("4g通话已开启")
#         else:
#             print("电量不足，无法使用4g通话，并让CPU以单核模式运行")
# phone_1 = Phone()
# phone_1.call_by_4g()
#
# # 类的继承
# class Phone2024(Phone): # 括号里写需要继承的类，单继承
#     face_id = "10000"
#
#     def call_by_5g(self):
#         print("2024年新功能：5g通话")
# phone_2 = Phone2024()
#
# phone_2.call_by_4g()
# print(phone_2.IMEI)
#
# class producer:
#     producer = "HW"
#
# class NFCReader:
#     nfc_type = "第五代"
#
#     def read_card(self):
#         print("NFC读卡")
#     def write_card(self):
#         print("NFC写卡")
# class RemoteControl:
#     rc_type = "红外遥控"
#
#     def control(self):
#         print("红外遥控开启了")
#
# #多继承
# # 若多继承的类有相同的属性，则最先继承的类的属性优先。
# # 例如上面的优先级是Phone2024 > NFCReader > RemoteConerol
# class Phone2077(Phone2024,producer,NFCReader,RemoteControl):
#     pass # 占位语句，表示空，没有内容的意思，用来保证语法完整性
#
# MyPhone = Phone2077()
# MyPhone.call_by_4g()
# MyPhone.call_by_5g()
# MyPhone.read_card()
# MyPhone.write_card()
# MyPhone.control()
# print(MyPhone.producer)
# import random
# import json
#
# var_1: int = 10
# var_2: float = 3.14
# var_3: str = 'Python'
# var_4: bool = True
#
# class Student:
#     pass
# stu: Student = Student()
#
# my_list: list = [1, 2, 3]
# my_tuple: tuple = (1, 'Python', True)
# my_dict: dict = {'a': 1, 'b': 2, 'c': 3}
#
# my_list: list[int] = [1, 2, 3]
# my_tuple: tuple[int, str, bool] = (1, 'Python', True)
# my_dict: dict[str, int] = {'a': 1, 'b': 2, 'c': 3}
#
# var_1 = random.randint(1,10) # type: int
# var_2 = json.loads('{"name": "John", "age": 25}') # type: dict[str,str]
# def func():
#     return 10
# var_3 = func() # type: int
# print(sum(map(int,str(123456))))
# print(eval('3+2'+'10'))
# len(zip([1,2,3], 'abcdefg'))
# x = {1,2,3}
# y = 3 * x
# x = zip('abc', '1234')
# x = list(range(10))
# print(x[-4:])
# x = [3,5,3,7]
# x.index(i)
# for i in x:
#     最优化理论 i==3:
#         print('1')
# index for index, value in enumerate([3,5,3,7]) 最优化理论 value==3
# my_set = { [1, 2, 3], [4, 5, 6] }
# print('abc' in 'abdc')
# names=['Bob','Tom','alice','jerry','Wendy','Smith']
# new_list=[name.upper() for name in names 最优化理论 len(name)>3]
# print(new_list)
# new_tuple=(x for x in range(31) 最优化理论 x%2==0)
# print(new_tuple)
from mpmath import mp
from tqdm import tqdm

# 设置精度
mp.dps = 1000


# 使用Chudnovsky算法计算圆周率
def chudnovsky(num_iterations):
    pi = mp.mpf(0)
    chunk_size = 1  # 每次计算的块大小
    num_chunks = num_iterations // chunk_size

    for chunk in tqdm(range(num_chunks)):
        pi_chunk = mp.mpf(0)
        for k in range(chunk * chunk_size, (chunk + 1) * chunk_size):
            numerator = (-1) ** k * mp.fac(6 * k) * (13591409 + 545140134 * k)
            denominator = mp.fac(3 * k) * mp.fac(k) ** 3 * 640320 ** (3 * k + 1.5)
            pi_chunk += mp.mpf(numerator) / mp.mpf(denominator)
        pi_chunk = pi_chunk * mp.mpf(12) / mp.mpf(640320 ** 1.5)
        pi_chunk = 1 / pi_chunk
        pi += pi_chunk

    return pi


# 设置迭代次数
num_iterations = 10000  # 调整此值以增加计算精度

# 计算圆周率并显示进度条
pi = chudnovsky(num_iterations)

# 打印结果
print("Pi:", pi)