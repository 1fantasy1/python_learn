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

# 写出提取正文内容的正则表达式，分析小说内容特点
# content_regx='<br />&nbsp;&nbsp;&nbsp;&nbsp;(.*?)<br />'

'''小说部分内容如下，
<br />&nbsp;&nbsp;&nbsp;&nbsp;有人告诉我“终有绿洲摇曳在沙漠”，但我清楚那是因为他们没有来过塔戈尔。
<br />&nbsp;&nbsp;&nbsp;&nbsp;这里是黄沙的世界，绵延的黄沙与天际相接，根本想象不出哪里才是沙的尽头。
<br />&nbsp;&nbsp;&nbsp;&nbsp;一沙一界，一界之内，一尘一劫。
<br />&nbsp;&nbsp;&nbsp;&nbsp;从出生到现在，我只知道有沙的地方就是我的故乡，而我也永远走不出故乡。
'''
# 内容特点分析
'''
    观察到文字包含 <br />&nbsp;&nbsp;&nbsp;&nbsp;和<br />之间，使用 (.*?)
    来匹配文字内容，其中'.'代表任意字符(除了换行),'*'代表任意多个字符,'?'代表非贪婪匹配，()将内容包含起来，使匹配结果仅包含括号里面的内容
'''

'''
    这是小说章节名称所在位置
    06.【美杜莎】<br>
'''
# 匹配出小说章节标题的正则表达式如下
# title_regx='(.*?)<br>'