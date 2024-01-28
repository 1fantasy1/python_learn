class Student():
    name = None
    age = None
    location = None
    def __init__(self, name, age, location):
        self.name = name
        self.age = age
        self.location = location
    def __str__(self):
        return f"Student类对象，name = {self.name},age = {self.age},location = {self.location}"
list_1 = []
for i in range(1,2):
    print(f"当前录入第{i}位学生信息，总共需录入10位学生信息")
    name = input("请输入学生姓名：")
    age = input("请输入学生年龄：")
    location = input("请输入学生地址：")
    stu = Student(name, age, location)
    list_1.append(stu)
    print(f"学生{i}信息录入完成，信息为：【学生姓名：{stu.name}，年龄：{stu.age}，地址：{stu.location}】")
for stu in list_1:
    print(stu)