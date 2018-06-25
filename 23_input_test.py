import numpy as np
import tensorflow as tf
import keras
import pickle
import re
import os
import time
from pickle import load
from numpy import array
from numpy import argmax
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
np.random.seed(1)

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('num_CV', 5, 'N-cross-validation')
tf.app.flags.DEFINE_string('word2vec_path', 'word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt', 'path to directory')
tf.app.flags.DEFINE_string('dataset_path', 'crossvalidation/newProverb2_jieba_v3_1-both.pkl', 'all dataset of path to directory')
tf.app.flags.DEFINE_string('train_path', 'crossvalidation/newProverb2_jieba_v3_1-1-train.pkl', 'train dataset of path to directory')
tf.app.flags.DEFINE_string('test_path', 'crossvalidation/newProverb2_jieba_v3_1-1-test.pkl', 'test dataset of path to directory')
tf.app.flags.DEFINE_integer('input_length', 10, 'Input sentence length')
tf.app.flags.DEFINE_integer('dim_size', 512, 'The Dimensions of Word2Vec')

# load a clean dataset
def load_sentences(filename):
        return load(open(filename, 'rb'))

def load_word2vec(wv_path):
    dim = 0
    words_vectors = {}
    words = []
    with open(wv_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 2:
                dim = int(tokens[1])
                continue
            word = tokens[0]
            words.append(word)
            vec = np.asarray([float(t) for t in tokens[1:]])
            words_vectors[word] = vec
    return words, words_vectors

# Compare dataset's word and word2vec's word
def compare_WV(dataset, wv):
    lines = list()
    train_wv = list()
    train_words = list()
    train_nonexist_wv = list()
    train_exist_wv = list()
    new_wv = dict()
    for i in range(len(dataset)):
        line = list()
        train_line = dataset[i, 1] + ' ' + dataset[i, 2]
        words = train_line.split(' ')
        for word in words:
            if word in wv and word not in train_exist_wv:
                train_exist_wv.append(word)
            if word not in wv and word not in train_nonexist_wv:
                train_nonexist_wv.append(word)
            if word not in train_words:
                train_words.append(word)
            if word in wv:
                line.append(word)
        lines.append(train_line)
        train_wv.append(' '.join(line))
    for word in train_exist_wv:
        new_wv[word] = wv[word]
    print('訓練集字詞量', len(train_words))
    print('訓練集存在於字詞向量長度', len(train_exist_wv))
    print('訓練集不存在於字詞向量長度', len(train_nonexist_wv))
    return train_wv, new_wv

def calculation_embedding_weights(words_vectors, tokenizer, vocab_size, dim_size):
    embedding_weights = np.zeros((vocab_size, dim_size))
    for word, vector in words_vectors.items():
        index = int(tokenizer.texts_to_sequences([word])[0][0])
        embedding_weights[index] = vector
    return embedding_weights

# fit a tokenizer
def create_tokenizer(lines):
        tokenizer = Tokenizer(oov_token='<UNK>')
        tokenizer.fit_on_texts(lines)
        return tokenizer

# max sentence length
def max_length(lines):
        return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    print(len(prediction))
    integers = [argmax(vector) for vector in prediction]
    print(integers)
    target = list()
    for i in integers:
            word = word_for_id(i, tokenizer)
            if word is None:
                    break
            target.append(word)
    return ' '.join(target)

def segment(sentence):
    print('Your input is ', sentence)
    words = list(jieba.cut(sentence, cut_all=False))
    #words = jieba.cut_for_search(sentence)
    print("Output 精確模式 Full Mode：")
    new_words = []
    #print(' '.join(words))
    a = False
    for word in words:
        if word == '-' or word == ' ':
            a= True
            continue
        new_words.append(word)
    print(' '.join(new_words))
    return new_words

def main(argv):
    # parameter
    # parameter
    wv_path = FLAGS.word2vec_path
    input_length = FLAGS.input_length
    dim_size = FLAGS.dim_size
    model_path = 'models/seq2seq/model.h5'

    #dataset parameter
    dataset_path = FLAGS.dataset_path
    train_path = FLAGS.train_path
    test_path = FLAGS.test_path

    # Load datasets
    dataset = load_sentences(dataset_path)
    train = load_sentences(train_path)
    test = load_sentences(test_path)

    # Load Word2Vec
    print('Loading Word2Vec ...')
    loadWV_s = time.time()
    # words_index, words_vectors = load_word2vec_pkl(pkl_path)
    words_index, words_vectors = load_word2vec(wv_path)
    loadWV_e = time.time()
    print('Finished Load elapsed time: %.2f sec.' % (loadWV_e - loadWV_s))

    # prepare train tokenizer
    sentences, words_vectors = compare_WV(dataset, words_vectors)
    #texts = [word for word in words_index]  
    train_tokenizer = create_tokenizer(sentences)
    train_vocab_size = len(train_tokenizer.word_index) + 1
    while True:
        input_sentence = input('Input a sentence:')
        if input_sentence == '0' or input_sentence == '': break
        sentence = [segment(input_sentence)]
        print(sentence)
        X = encode_sequences(train_tokenizer, input_length, sentence)
        print(X)

if __name__ == '__main__':
    tf.app.run()