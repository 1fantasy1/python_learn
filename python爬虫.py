import requests
from bs4 import BeautifulSoup
i = 1
for start_num in range(0,250,25):
    head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
    r = requests.get(f'https://movie.douban.com/top250?start={start_num}',headers=head)
    soup = BeautifulSoup(r.text,'html.parser')
    all_name = soup.findAll('span', attrs={'class':'title'})
    for name in all_name:
        title = name.string
        if "/" not in title:
            print(f"{i}.{title}")
            i += 1
# print(r.status_code)
# print(r.text)