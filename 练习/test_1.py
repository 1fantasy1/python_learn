import requests

# 定义Gitee文件的raw URL
raw_file_url = "https://gitee.com/fantasy_9928/python_learn/raw/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%B1%82%E4%B9%8B%E4%B8%8D%E5%BE%97%E8%A1%A8.txt"

# 下载文件内容
file_response = requests.get(raw_file_url)

# 检查请求是否成功
with open("downloaded_file.txt", "wb") as file:
    file.write(file_response.content)
    print("文件下载完成")