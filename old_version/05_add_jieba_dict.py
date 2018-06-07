import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora.dictionary import Dictionary
import re
import warnings
from pickle import dump
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')

dict_path = r'tools\extra_dict\dict.txt'
dict_path2 = r'tools\extra_dict\dict2.txt'
dict_path3 = r'tools\extra_dict\dict3.txt'
wv_path = r'word2vec\wiki100.model.bin'
dict1 = {}
dict2 = {}
dict3 = {}

with open(dict_path, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip().split(' ')[0]
        count = int(line.strip().split(' ')[1])
        dict1[word] = count
        dict3[word] = count

print('Load Word2Vec')
word2vectors = KeyedVectors.load_word2vec_format(wv_path, binary = True)
print('Load Finished')


for word in word2vectors.vocab.keys():
    vocab_word = word2vectors.vocab[word]
    word_freq = 5 # vocab_word.count
    if word not in dict1:
        dict3[word] = word_freq
    else:
        dict3[word] += dict1[word]+word_freq
    dict2[word] = word_freq
'''
with open(dict_path2, 'w', encoding='utf-8') as f:
    for word in word2vectors.vocab.keys():
        word_freq = word2vectors.vocab[word].count
        f.write("%s %d\n" % (word, 1))

with open(dict_path3, 'w', encoding='utf-8') as f:
    for word in dict3:
        f.write("%s %d\n" % (word, dict3[word]))
        # print("[%s] => %d" % (word, dict3[word]))
'''
print('Finished')

print(len(dict1))
print(len(dict2))
print(len(dict3))


    

