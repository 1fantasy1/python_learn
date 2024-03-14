list_1 = []
counts = {}
num_1 = input("请输入一些数字，用空格分开：").split()
num_1 = [int(num) for num in num_1]
for num in num_1:
    counts[num] = counts.get(num,0) + 1
for num, count in counts.items():
    if count == 1:
        list_1.append(num)
print("s",list_1)