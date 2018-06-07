import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora.dictionary import Dictionary
import re
import warnings
import pickle
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')

def create_dictionaries(wv):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(wv.vocab.keys(), allow_update=True)
    w2indx = {key: index + 1 for index, key in gensim_dict.items()}
    w2vec = {word: wv[word] for word in w2indx.keys()}
    return w2indx, w2vec

# 996,819 words
word_vectors = KeyedVectors.load_word2vec_format('wiki300.model.bin', binary = True)
index_dict, word_vectors= create_dictionaries(word_vectors)

with open('Word2Vec_Dict.pkl', 'wb') as f:
    pickle.dump(index_dict, f)
    pickle.dump(word_vectors, f)
