import requests
from bs4 import BeautifulSoup
from hanziconv import HanziConv as HC
import re

def processText(html):
    text = str(html)
    line = ''
    R = False
    if  '\n' in text:
        return R, text
    text = text.strip()
    text = HC.toTraditional(text)
    rule = ['-', '—']
    for r in rule:
        if r in text:
            R = True
    text = re.split(r'\-+|—+', text)
    return R, text

def writeFile(data):
    path = './new.csv'
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            t = "\"%s\",\"%s\"\n" % (d[0], d[1])
            f.write(t)

url_web = 'http://www.haha365.com/xhy/'
htmlQ = 180
index_i = 1

data = []
for index in range(index_i, htmlQ, 1):
    page = "%sindex_%s.htm" % (url_web, index)
    print(page)
    response  = requests.get(page)
    if response .status_code == requests.codes.ok:
        response .encoding = 'gbk'
        soup = BeautifulSoup(response .text, 'html.parser')
        stories = soup.find_all('div', id='endtext')
        for s in stories:
            R, text = processText(s.text)
            if R:
                if len(text) != 2:
                    continue
                data.append(text)

writeFile(data)
