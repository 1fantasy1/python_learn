import re
import requests
from bs4 import BeautifulSoup
def title_url(all_url):
    """
    :param all_url: 需要解析的小说目录
    :return: 解析后的小说章节网址
    """
    head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
                                                                # 设置用户代理，绕过目录网页爬虫判定
    read = requests.get(all_url)                                # 读取目录网页内容传入read参数。
    read.encoding = "GBK"                                       # 将需读取网页设置编码为GBK
    soup = BeautifulSoup(read.text, features='html.parser')     # 第一个形参为需解析网页，第二个形参为需解析的为HTML网页。
    all_text = soup.findAll('a')                                # 使用FindAll方法寻找<a>标签下的各章节网址，传入all_text列表
    list_suffix = []     # 定义空列表准备传入各章节的网址
    list_all = []
    for text in all_text:
        """
        遍历all_text列表，整合网址传入列表list_1中。
        """
        f = text.get("href")    # 获取"<href>"标签下内容
        if re.search("/20_20386/(.*?).html", f, flags=0):# 查找到对应章节的网址后缀
            list_suffix.append(f)    # 追加网址到列表list_suffix中
    for i in list_suffix:
        website = f'https://www.biquge.co{i}'   # 拼接网址后缀
        list_all.append(website)    # 将拼接的网址传入列表list_all中
    return list_all

# def title_name()

if __name__ == '__main__':      # 测试函数是否可用
    title_url("https://www.biquge.co/20_20386/")