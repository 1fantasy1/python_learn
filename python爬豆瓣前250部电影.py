import requests  # 导入请求库
from bs4 import BeautifulSoup  # 导入BeautifulSoup库用于解析HTML

i = 1  # 初始排名

# 遍历每一页，每页包含25部电影
for start_num in range(0, 250, 25):
    # 添加头信息，模拟浏览器访问
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
    }
    # 发送GET请求获取豆瓣电影Top250页面
    r = requests.get(f'https://movie.douban.com/top250?start={start_num}', headers=head)
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(r.text, 'html.parser')
    # 查找所有包含电影标题的<span>标签，class为'title'
    all_name = soup.findAll('span', attrs={'class': 'title'})

    # 遍历每个电影标题
    for name in all_name:
        title = name.string  # 获取电影标题
        if "/" not in title:  # 如果标题中不包含'/'符号（排除多语言标题）
            print(f"{i}.{title}")  # 打印电影排名和标题
            i += 1  # 排名递增