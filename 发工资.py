wage = 10000
for i in range(1,21):
    import random
    performance = random.randint(1,10)
    if wage == 0:
        break
    if performance >= 5:
        wage -= 1000
        print(f"向员工{i}发放工资1000元，账户余额还剩{wage}元。")
    else:
        print(f"员工{i}，绩效分{performance},低于5，不发工资，下一位。")
print("工资发完了，下个月领取吧。")