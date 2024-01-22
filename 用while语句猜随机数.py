import random
size = int(input("请输入你要猜的大小："))
num = random.randint(1,size)
i = 1
num_1 = int(input("请输入你猜的数："))
while num_1 != num:
    print("你猜错了")
    if num_1 > num:
        print("你猜的数大了")
    else:
        print("你猜的数小了")
    num_1 = int(input("请重新输入："))
    i += 1
print(f"你猜对了,总共猜了{i}次。")