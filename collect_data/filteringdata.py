import re

def writeFile(D, start, limit):
    path = './newProverb2_v2_0.csv'
    with open(path, 'w', encoding='utf-8') as f:
        for index, d in enumerate(D):
            index += 1
            if index < start: continue
            if index == limit: break
            f.write(',%s,%s\n' % (d[1], d[2]))

def expansion(arr):
    d = []
    index = 1
    if bool(re.search('；', arr[1])):
        arr2 = re.splie('；', arr[1])
        for v in arr2:
            d.append([])

def LoadData(path):
    datas = []
    i = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if bool(re.search('（|）', line)):
                continue
            arr = re.split(',', line)
            if (bool(re.search('；', arr[1])) and bool(re.search('；', arr[2]))) or (len(arr) != 3):
                continue
            if bool(re.search('；', line)):
                #data.append(expansion(arr))
                continue
            datas.append(arr)
            i += 1
        print(i)
    return datas

file_path = './newProverb2_v0.csv'
datas = LoadData(file_path)
writeFile(datas, 300, 1005)
print(len(datas))
