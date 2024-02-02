# import re
import sys
import requests
from bs4 import BeautifulSoup

# proxies = [
#     {'http':'socks5://127.0.0.1:1234'},
#     {'https':'socks5://127.0.0.1:1234'}
# ]                                             # 如果网站被墙，则挂代理
# proxies = random.choice(proxies)
# print(proxies)

class downloader(object):
    def __init__(self):
        self.server = "https://www.biquge.co/"          # 爬取的网站
        self.all_url = "https://www.biquge.co/20_20386/" # 爬取的小说目录
        self.name_all = []      # 存放小说目录
        self.list_all = []      # 存放章节链接
        self.num = 0            # 章节数

    def title_url(self):
        """

        :param all_url: 需要解析的小说目录
        :return: 解析后的小说章节网址
        """
        head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
            # 设置用户代理，绕过目录网页爬虫判定
        read = requests.get(self.all_url)       # 读取目录网页内容传入read参数。
        read.encoding = "GBK"       # 将需读取网页设置编码为GBK
        soup = BeautifulSoup(read.text, features='html.parser')  # 第一个形参为需解析网页，第二个形参为需解析的为HTML网页。
        all_text = soup.findAll('a')    # 使用FindAll方法寻找<a>标签下的各章节网址，传入all_text列表
        self.num = len(all_text[41:1690])    # 计算章节数
        for each in all_text[41:1690]:
            self.name_all.append(each.string)                       #获取章节名
            self.list_all.append(self.server + each.get('href'))    #拼接网址

    def get_text(self,url_1):
        """

        :param url_1: 章节网址
        :return: 正文内容
        """
        r = requests.get(url_1)  # 将获取的网址传入
        r.encoding = "GBK"  # 将解析的网页设置编码为GBK
        soup = BeautifulSoup(r.text, 'html.parser')  # 使用bs4进行HTML处理
        texts = soup.find_all('div', id='content')  # 根据网页格式找到文本正文内容
        texts = texts[0].text.replace('\r\n', '')  # 去除文本中无效的内容
        return texts
    def write_text(self, name, path, text):
        """

        :param name:    章节名
        :param path:    文件保存路径
        :param text:    正文内容
        :return:        无
        """
        write_flag = True
        with open(path, 'a', encoding='utf-8') as f:
            f.write(name + '\n')        # 写入章节名
            f.writelines(text)          # 写入正文
            f.writelines('\n')          # 换行隔开每一章
if __name__ == '__main__':
    dl = downloader()
    dl.title_url()
    print("斗破苍穹开始下载：")
    for i in range(dl.num):
        g = dl.list_all[i]
        dl.write_text(name = dl.name_all[i], path= "D:/斗破苍穹.txt", text = dl.get_text(g))
        sys.stdout.write("已下载：%.3f%%" % float(i/dl.num) + "\r")
        sys.stdout.flush()
    print("《斗破苍穹》下载完成")


# 废案

# url_1 = title_url_acquire.title_url('https://www.biquge.co/20_20386/')
# text_1 = text_acquire.get_text(url_1[9])
# print(text_1)

        # list_suffix = []        # 定义空列表准备传入各章节的后缀
        # list_all = []           # 定义空列表准备传入各章节的网址
        # for text in all_text:
        #     """
        #     遍历all_text列表，整合网址传入列表list_1中。
        #     """
        #     f = text.get("href")  # 获取"<href>"标签下内容
        #     if re.search("/20_20386/(.*?).html", f, flags=0):  # 查找到对应章节的网址后缀
        #         list_suffix.append(f)  # 追加网址到列表list_suffix中
        # for i in list_suffix:
        #     website = f'https://www.biquge.co{i}'  # 拼接网址后缀
        #     list_all.append(website)  # 将拼接的网址传入列表list_all中
