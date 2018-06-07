import requests
from bs4 import BeautifulSoup
import re

url_web = 'http://www.haha365.com/xhy/'

def processText(html):
    line = ''
    text = str(html)
    R = False
    if (not '\n' in text) and (('—' in text) or (('-' in text))):
        R = True
    if R:
        text = text.strip()
        rule = re.compile(r"[^\u4e00-\u9fa5|—+|\-+]")
        print(rule)
        line = rule.sub('', text)
    return R, line

q = 180
index = 1
i = 1
for index in range(q):
    index += 1
    page = "%sindex_%s.htm" % (url_web, index)
    print(page)
    r = requests.get(page)
    if r.status_code == requests.codes.ok:
        r.encoding = 'gbk'
        soup = BeautifulSoup(r.text, 'html.parser')
        stories = soup.find_all('div', id='endtext')
        for s in stories:
            R, text = processText(s.text)
            if R:
                print(i, "\t", text)
                i += 1
