from bs4 import BeautifulSoup
import requests

# 循环遍历每一页，每页包含25部电影信息
for start_num in range(0, 250, 25):
    # 添加头信息，模拟浏览器访问
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
    }
    # 发送GET请求获取豆瓣电影Top250页面
    response = requests.get(f'https://movie.douban.com/top250?start={start_num}', headers=head)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到所有class为‘item’的div标签，每个标签代表一部电影
    movies = soup.find_all('div', class_='item')

    # 遍历每部电影，提取排名、标题、评分和链接信息并打印出来
    for movie in movies:
        # 提取电影排名
        rank = movie.find('em').text
        # 提取电影标题
        title = movie.find('span', class_='title').text
        # 提取电影评分
        rating = movie.find('span', class_='rating_num').text
        # 提取电影链接
        link = movie.find('a')['href']

        # 打印电影信息
        print(f"{rank}- {title}- {rating}- {link}")
