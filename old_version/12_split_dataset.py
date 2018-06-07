from pickle import load, dump
from numpy.random import rand
from numpy.random import shuffle
import numpy as np
np.random.seed(1)

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('./pkl_data/newProverb2_jieba_v2_1.pkl')
n_sentences = len(raw_dataset)
train_size = n_sentences * 0.8
train_size = int(train_size)
dataset = raw_dataset
shuffle(dataset)
train, test = dataset[:train_size], dataset[train_size:]
save_clean_data(dataset, './crossvalidation/newProverb2_jieba_v2_1-both.pkl')
save_clean_data(train, './crossvalidation/newProverb2_jieba_v2_1-train.pkl')
save_clean_data(test, './crossvalidation/newProverb2_jieba_v2_1-test.pkl')
