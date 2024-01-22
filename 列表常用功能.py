age = [21,25,21,23,22,20]#定义一个列表
print(age)

age.append(31)#追加一个数字到列表尾部
print(age)

age.extend([29,33,30])#追加一个列表到列表尾部
print(age)

print(age[0])#取出第一个元素

print(age[-1])#取出最后一个元素

sub = age.index(31)
print(sub)
