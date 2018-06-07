import numpy as np
import tensorflow as tf
import keras
import pickle
import re
import os
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
np.random.seed(1)

# load a clean dataset
def load_sentences(filename):
        return load(open(filename, 'rb'))

# load Word2Vec
def load_word2vec(pkl_path):
    with open(pkl_path, 'rb') as f:
        words_index = load(f)
        words_vectors = load(f)
    return words_index, words_vectors

def compare_WV(train, wv):
    lines = []
    vocab = []
    new_wv = dict()
    for i in range(len(train)):
        line = []
        train_line = train[i, 1] + ' ' + train[i, 2]
        words = train_line.split(' ')
        for word in words:
            if word in wv and word not in vocab:
                vocab.append(word)
            if word in wv:
                line.append(word)
        lines.append(' '.join(line))
    for word in vocab:
        new_wv[word] = wv[word]
    tt = np.array(new_wv)
    return lines, new_wv

def calculation_embedding_weights(words_vectors, tokenizer, vocab_size, dim_size):
    embedding_weights = np.zeros((vocab_size, dim_size))
    for word, vector in words_vectors.items():
        index = tokenizer.texts_to_sequences([word])
        embedding_weights[index, :] = vector
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

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
        actual, predicted = list(), list()
        for i, source in enumerate(sources):
                if i ==1: break
                # translate encoded source text
                ll = []
                for i in source:
                        ll.append(word_for_id(i, tokenizer))
                print(ll)
                source = source.reshape((1, source.shape[0]))
                translation = predict_sequence(model, tokenizer, source)
                label, raw_src, raw_target = raw_dataset[i]
                if i < 10:
                        print('i=[%s], src=[%s], target=[%s], predicted=[%s]' % (i, raw_src, raw_target, translation))
                actual.append(raw_target.split())
                predicted.append(translation.split())
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# parameter
file_path = 'newProverb2_jieba_v1_1.csv'
pkl_path = 'word2vec/300Word2Vec_Dict.pkl'
input_length = 15
dim_size = 300
units = 256
batch_size=30
epochs=20
model_path = 'models/seq2seq/model.h5'

# load datasets
dataset = load_sentences('./crossvalidation/newProverb2_jieba_v2_3-both.pkl')
train = load_sentences('./crossvalidation/newProverb2_jieba_v2_3-train.pkl')
test = load_sentences('./crossvalidation/newProverb2_jieba_v2_3-test.pkl')

# load Word2Vec
print('Loading Word2Vec and Dict ...')
words_index, words_vectors = load_word2vec(pkl_path)
print('Finished Load')

# prepare train tokenizer
token_train, words_wv = compare_WV(train, words_vectors)
train_tokenizer = create_tokenizer(token_train)
train_vocab_size = len(train_tokenizer.word_index) + 1
X_length = max_length(dataset[:, 1])
Y_length = max_length(dataset[:, 2])

# prepare data
trainX = encode_sequences(train_tokenizer, input_length, train[:, 1])
testX = encode_sequences(train_tokenizer, input_length, test[:, 1])

# load model
model = load_model(model_path)
# test on some training sequences
print('train')
evaluate_model(model, train_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, train_tokenizer, testX, test)
