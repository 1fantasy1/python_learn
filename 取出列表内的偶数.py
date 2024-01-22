def even_1():
    list = [1,2,3,4,5,6,7,8,9,10]
    list_1 = []
    i = None
    while i < len(list):
        shu = list[i]
        if shu % 2 == 0:
            list_1.append(shu)
        i += 1
    print(f"通过while循环,从列表:{list}中取出偶数,组成的新列表:{list_1}")

def even_2():
    list = [1,2,3,4,5,6,7,8,9,10]
    list_2 = []
    for j in list:
        if j % 2 == 0:
            list_2.append(j)
    print(f"通过for循环,从列表:{list}中取出偶数,组成的新列表:{list_2}")
even_1()
even_2()