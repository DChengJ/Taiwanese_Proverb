import requests
from bs4 import BeautifulSoup
import re

url_web = 'http://www.ctas.tc.edu.tw/index2.php/files/inp51/index2.php?content=life&nsid=xiehouyu&id='
htmlQ=14036

def processStr(html):
    text = str(html).strip()
    return text

def writeFile(data):
    path = './new2.csv'
    with open(path, 'a', encoding='utf-8') as f:
        for d in data:
            t = ",\"%s\",\"%s\"\n" % (d[0], d[1])
            f.write(t)

data = []

for index in range(1, htmlQ+1, 1):
    page = "%s%s" % (url_web, index)
    response  = requests.get(page, verify=False, timeout=30)    
    if response .status_code == requests.codes.ok:
        response .encoding = 'utf-8'
        soup = BeautifulSoup(response .text, 'html.parser')
        stories = soup.find_all('td', class_='px15 , ls2')
        if len(stories) != 2:
            print("!!", page)
            continue
        t1, t2= processStr(stories[0].text), processStr(stories[1].text)
        data.append([t1, t2])
    if index % 200 == 0:
        print(page)
        writeFile(data)
