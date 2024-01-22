name = input("请输入你的语句：")
let = input("请输入你要统计的字母：")
a_sum = 0
for i in name:
    if i == let:
        a_sum += 1
print(f"{let}的总和是：{a_sum}")