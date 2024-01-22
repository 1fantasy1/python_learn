num = int(input("请输入你要求1到几的偶数："))
eve = 0
for i in range(1,num):
    if i % 2 == 0:
        eve += 1
print(eve)