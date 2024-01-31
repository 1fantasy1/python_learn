import requests
head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
r = requests.get('https://movie.douban.com/top250',headers=head)
print(r.status_code)
print(r.text)