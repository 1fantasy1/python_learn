import requests
from bs4 import BeautifulSoup
head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
r = requests.get('https://movie.douban.com/top250',headers=head)
soup = BeautifulSoup(r.text,'html.parser')
all_name = soup.findAll('span', attrs={'class':'title'})
for i in all_name:
    print(i.string.strip(" / "))

# print(r.status_code)
# print(r.text)