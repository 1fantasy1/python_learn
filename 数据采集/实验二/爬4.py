from bs4 import BeautifulSoup
import requests
from tqdm import tqdm  # 导入 tqdm

# 打开文件，准备写入电影信息
with open('movies.txt', 'w', encoding='utf-8') as f:
    # 使用 tqdm 来包装循环，显示进度条
    for start_num in tqdm(range(0, 250, 25), desc="Progress"):
        # 添加头信息，模拟浏览器访问
        head = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
        }
        # 发送GET请求获取豆瓣电影Top250页面
        response = requests.get(f'https://movie.douban.com/top250?start={start_num}', headers=head)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 找到所有class为‘item’的div标签，每个标签代表一部电影
        movies = soup.find_all('div', class_='item')

        for movie in movies:
            rank = movie.find('em').text
            title = movie.find('span', class_='title').text
            rating = movie.find('span', class_='rating_num').text
            link = movie.find('a')['href']

            # 获取详细信息
            detail_url = link
            detail_response = requests.get(detail_url, headers=head)
            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

            # 获取导演
            director = detail_soup.find('a', rel='v:directedBy').text
            # 获取编剧
            writers = [a.text for a in detail_soup.find_all('a', rel='v:writer')]
            writers = ' / '.join(writers)
            # 获取主演
            actors = [a.text for a in detail_soup.find_all('a', rel='v:starring')]
            actors = ' / '.join(actors)
            # 获取类型
            genres = [a.text for a in detail_soup.find_all('span', property='v:genre')]
            genres = ' / '.join(genres)
            # 获取上映时间
            release_date = detail_soup.find('span', property='v:initialReleaseDate').text
            # 获取片长
            runtime = detail_soup.find('span', property='v:runtime').text
            # 获取评分人数
            rating_count = detail_soup.find('span', property='v:votes').text
            # 获取剧情简介
            summary = detail_soup.find('span', property='v:summary').text

            # 将电影信息写入文件
            f.write(f"{rank}- {title}- {rating}- {link}\n")
            f.write(f"导演: {director}\n")
            f.write(f"编剧: {writers}\n")
            f.write(f"主演: {actors}\n")
            f.write(f"类型: {genres}\n")
            f.write(f"上映时间: {release_date}\n")
            f.write(f"片长: {runtime}\n")
            f.write(f"评分人数: {rating_count}\n")
            f.write(f"剧情简介: {summary}\n")
            f.write("\n")
