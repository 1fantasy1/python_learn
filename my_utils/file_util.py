def print_file_info(file_name):
    """
    将给定路径文件输出
    :param file_name:即将读取的文件路径
    :return: None
    """
    i = None
    try:
        i = open(file_name, 'r')
        print(f"文件的全部内容如下{i.read()}")
    except Exception as e:
        print(f"程序出现异常了，原因是：{e}")
    finally:
        if i is not None: # 如果变量不是None，则关闭文件
            i.close()
def append_to_file(file_name, data):
    """
    追加给定路径文件内容
    :param file_name:即将读取的文件路径
    :param data: 需要追加的内容
    :return: None
    """
    i = open(file_name, 'a')
    i.write(data)
    i.write('\n')
    i.close()

if __name__ == '__main__':
    print_file_info('D:/测试.txtxxx')
if __name__ == '__main__':
    append_to_file("D:/测试.txt","789")