# 设计一个手机类
class Phone:
    __is_5g_enable = False # 5g状态

    #提供私有成员方法：__check_5g
    def __check_5g(self):
        if self.__is_5g_enable == True:
            print("5g开启")
        else:
            print("5g关闭，使用4g网络")
    #提供公开成员方法：__call_by_5g
    def call_by_5g(self):
        self.__check_5g()
        print("正在通话中")
# 创建一个手机对象
phone = Phone()
# 使用内置公开成员方法
phone.call_by_5g()