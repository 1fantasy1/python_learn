import random
name = input("请输入您的姓名，获得随机金额：")
money = random.randint(50000,10000000)
print("太棒了，你获得了一些钱。")
def query(show):
    if show:
        print("----------查询余额----------")
    print(f"{name}，您好，您的余额剩余：{money}元。")
def sav():
    print("--------------存款--------------")
    i = int(input("请输入您要存款的金额；"))
    global money
    if i < 0:
        print("存的金额不能小于0哦。")
        sav()
    else:
        money += i
        print(f"{name}，您好，您存款{i}元成功。")
        #调用query函数查询余额
        query(False)
def take():
    print("--------------取款--------------")
    i = int(input("请输入您要取款的金额；"))
    global money
    if i > money:
        print(f"您的余额不足，您当前的余额是{money}")
        take()
    elif i < 0:
        print("取的金额不能小于0哦。")
        take()
    else:
        money -= i
        print(f"{name}，您好，您取款{i}元成功。")
        # 调用query函数查询余额
        query(False)
def menu():
    print("--------------主菜单--------------")
    print(f"{name}，您好，欢迎来到沧海ATM，请选择操作：")
    print("查询余额\t[输入1]")
    print("存款\t\t[输入2]")
    print("取款\t\t[输入3]")
    print("退出\t\t[输入4]")
    return input("请输入您的选择：")
while True:
    keyboard_input = menu()
    if keyboard_input == "1":
        query(True)
        continue
    elif keyboard_input == "2":
        sav()
        continue
    elif keyboard_input == "3":
        take()
        continue
    elif keyboard_input != "4":
        print("您没有输入正确的数字，请重新输入！")
        continue
    else:
        print("结束了！")
        break