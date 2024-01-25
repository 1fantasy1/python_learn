f = open("D:/bill.txt","r",encoding="utf-8")
k = open("D:/bill.txt.bak","a",encoding="utf-8")
for line in f:
    line = line.strip("\n") # 去除开头结尾换行符
    list_1 = line.split("，")
    if list_1[4] == "正式":
        k.write(line)
        k.write("\n")
f.close()
k.close()