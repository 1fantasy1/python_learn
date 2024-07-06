from bs4 import BeautifulSoup
html = """<html>
<head>
<title>页面标题</title>
</head>
<body>
<img src="image1.jpg" alt="Image 1">
<img src="image2.jpg" alt="Image 2">
<img src="image3.jpg" alt="Image 3">
</body>
</html>
"""

def extract_image_links(html):
    # 用BeautifulSoup解析HTML
    soup = BeautifulSoup(html, 'html.parser')

    # 找到所有的图片标签
    img_tags = soup.find_all('img')

    # 提取每个图片标签的src属性值
    image_urls = []
    for img in img_tags:
        image_urls.append(img['src'])

    return image_urls

print(extract_image_links(html))