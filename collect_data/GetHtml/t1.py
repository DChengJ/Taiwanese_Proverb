import requests
from bs4 import BeautifulSoup
import re

# 下載 Yahoo 首頁內容
r = requests.get('http://www.haha365.com/xhy/index_180.htm')

def processText(html):
    print('--', str(html))

# 確認是否下載成功
if r.status_code == requests.codes.ok:
  # 以 BeautifulSoup 解析 HTML 程式碼
  r.encoding = 'gbk'
  soup = BeautifulSoup(r.text, 'html.parser')

  # 以 CSS 的 class 抓出各類頭條新聞
  stories = soup.find_all('div', class_='cat_llb')
  print(type(stories))
  i = 1
  for s in stories:
    processText(s.text)
