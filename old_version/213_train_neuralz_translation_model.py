from pickle import load
from numpy import array
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# Load a Clean Dataset
def load_dataset(file_path):
    return load(open(file_path, 'rb'))

def create_tokenizer(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def max_length(sentences):
    return max(len(sentence.split()) for sentence in sentences)
'''
def load_word2vec(wv_path):
    embeddings_index = {}
    with open(wv_path, 'rb') as f:
'''
# load datasets
dataset = load_dataset('./crossvalidation/newProverb2_jieba_v2_3-both.pkl')
train = load_dataset('./crossvalidation/newProverb2_jieba_v2_1-train.pkl')
test = load_dataset('./crossvalidation/newProverb2_jieba_v2_1-test.pkl')
pkl_path = 'word2vec/300Word2Vec_Dict.pkl'

# Load NLP Data
#print('Loading Word2Vec and Dict ...')
with open(pkl_path, 'rb') as f:
    index_dict = load(f)
    word_vectors = load(f)
#print('Finished Load')

lines = []
all_words = []
nonexistents = []
nonseq = []

for i in range(len(dataset)):
    # print(dataset[i, 1] + ' ' + dataset[i, 2])
    lines.append(dataset[i, 1] + ' ' + dataset[i, 2])


for line in lines:
    words = line.split(' ')
    Exist = False
    for word in words:
        if word not in index_dict:
            Exist = True
        if word not in all_words and word not in all_words:
            all_words.append(word)
        if word not in index_dict  and word not in nonexistents:
            nonexistents.append(word)
    if Exist:
        nonseq.append(line)
            
print('dataset len:', len(lines))
print('All words: ', len(all_words))
print('Sequence Counts: ', len(nonseq))
print('No Exist Words: ', len(nonexistents))
'''
with open('nonexistent.txt', 'w') as f:
    for word in nonexistents:
        f.write(word + '\n')
'''
'''
for line in nonexistent:
    print(line)
    
print(len(lines))    
print(len(nonexistent))
'''
'''
dd = np.concatenate((dataset[:, 1], dataset[:, 2]), axis=0)

i = 0
for line in dd:
    line = line.split(' ')
    for word in line:
        if word not in all_words:
            all_words.append(word)
        if word not in index_dict and word not in nonexistent:
            nonexistent.append(word)
            i+=1
            break
'''
'''
with open('nonexistent.txt', 'w') as f:
    for word in nonexistent:
        f.write(word + '\n')
'''
