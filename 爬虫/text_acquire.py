import requests
from bs4 import BeautifulSoup

def get_text(title_url):
    r = requests.get(title_url) # 将获取的网址传入
    r.encoding = "GBK"  # 将解析的网页设置编码为GBK
    soup = BeautifulSoup(r.text, 'html.parser') # 使用bs4进行HTML处理
    texts = soup.find_all('div',id='content') # 根据网页格式找到文本正文内容
    texts_new = texts[0].text.replace('\r\n','  ') # 去除文本中无效的内容
    return texts_new
if __name__ == '__main__':
    get_text('https://www.biquge.co/20_20386/16655814.html') # 对函数进行测试