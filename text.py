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
# if age >= 18:#冒号不能忘
#     print("您已成年，游玩需要补票10元。")
# print("祝您游玩愉快。")
# print("欢迎来到沧海动物园。")
# text_height = int(input("请输入你的身高(cm)："))
# if text_height > 120:
#     print("您的身高超出120cm，游玩需要补票10元。")
# else:
#     print("您的身高未超出120cm，可以免费游玩。")
# print("祝您游玩愉快。")
# num = int(input(""))
# if int(input("请输入第一次猜想的数字：")) == num:
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
yuan_1 = (1,2,3,4,5,6,7,8,9)
yuan_2 = (1, )#元组只有一个元素后面要加逗号.
yuan_3 = ((1,2),(3,4))
print(f"{type(yuan_1)},{type(yuan_2)},{type(yuan_3)}")
num = yuan_3[1][1]
print(num)
"""
!!!
元组只有index(),count(),len(元组)
!!!
"""

#元组不可修改(如增加或删除元素),但有特殊情况,元组内有可修改的数据容器就可修改
yuan_4 = (1,2,[3,4])
yuan_4[2][1] = 5
print(yuan_4)