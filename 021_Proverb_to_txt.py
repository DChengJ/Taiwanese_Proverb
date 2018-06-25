import re

data_path = 'word2vec_data/newProverb2_jieba_v3_1.csv'
output_path = 'word2vec_data/newProverb2_jieba_v3_1.txt'

sentences = ''

with open(output_path, 'w', encoding='utf-8') as fw:
	with open(data_path, 'r', encoding='utf-8') as fr:
		count = 0
		for line in fr:
			line = line.strip()
			pairs = re.split(r'\",\"', line[1:-1])
			text = pairs[1] + ' ' + pairs[2]
			count += len(text.split(' '))
			fw.write(text + '\n')
		print(count)