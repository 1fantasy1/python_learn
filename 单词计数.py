# f = open("D:/word.txt","r",encoding="utf-8")
# content = f.read()
# print(content)
# count = content.count("itheima")
# print(f"itheima出现的次数为：{count}")
# f.close()

f = open("D:/word.txt","r",encoding="utf-8")
i = 0
for line in f:
    line = line.strip()# 去除开头结尾的换行符
    words = line.split(" ")# 以空格切分
    for word in words:
        if word == "itheima":
            i += 1
print(f"itheima出现的次数为：{i}")
