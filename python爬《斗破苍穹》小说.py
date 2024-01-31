import re
import requests
import random
from bs4 import BeautifulSoup
# proxies = [
#     {'http':'socks5://127.0.0.1:1234'},
#     {'https':'socks5://127.0.0.1:1234'}
# ]
# proxies = random.choice(proxies)
# print(proxies)
# url = 'https://www.biquge.co/20_20386/'
# try:
#     response = requests.get(url) #不使用代理
#     print(response.status_code)
#     if response.status_code == 200:
#         print(response.text)
# except requests.ConnectionError as e:
#     print(e.args)


# for start_num in range(0,250,25):
# /5594096.html
head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
read = requests.get(f'https://www.biquge.co/20_20386/') # 将需解析的网页传入read参数。
read.encoding = "GBK" # 将解析的网页设置编码为GBK
soup = BeautifulSoup(read.text,features='html.parser') # 第一个形参为需解析网页，第二个形参为解析HTML网页。
all_text = soup.findAll('a')
# print(all_text)
list_1 = []
for text in all_text:
    f = text.get("href")
    if re.search("/20_20386/(.*?).html",f,flags=0):
        list_1.append(f)
print(list_1)

# print(all_text)
    # for name in all_name:
    #     title = name.string
    #     if "/" not in title:
    #         print(f"{i}.{title}")
    #         i += 1