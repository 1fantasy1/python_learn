import random
num = random.randint(1,50)
num_1 = int(input("请输入你猜的数"))
if  num_1 == num:
    print("你猜对了")
else:
    print("你猜错了")
    if num_1 > num:
        print("大了")
    else:
        print("小了")
    num_2 = int(input("再猜第二次"))
    if  num_2 == num:
            print("你猜对了")
    else:
            print("你猜错了")
            if num_2 > num:
                print("大了")
            else:
                print("小了")
            num_3 = int(input("再猜最后一次"))
            if  num_3 == num:
                    print("你猜对了")
            else:
                    print(f"你猜错了，答案是{num}")