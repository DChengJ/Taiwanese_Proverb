import numpy as np
import tensorflow as tf
import keras
import pickle
import re
import os
from pickle import load
from numpy import array
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
np.random.seed(1)

# load dataset
def load_sentences(data_path):
    return load(open(data_path, 'rb'))

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

# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

# define NMT model
def define_model(vocab, dim_size, timesteps, n_units, embedding_weights):
    model = Sequential()
    model.add(Embedding(input_dim=vocab, output_dim=dim_size, input_length=timesteps, weights=[embedding_weights], mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab, activation='softmax')))
    return model

# parameter
file_path = 'newProverb2_jieba_v1_1.csv'
pkl_path = 'word2vec/300Word2Vec_Dict.pkl'
input_length = 15
dim_size = 300
units = 256
batch_size=30
epochs=50
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

# prepare embedding weight
embedding_weights = calculation_embedding_weights(words_wv, train_tokenizer, train_vocab_size, dim_size)
print(len(embedding_weights))

# prepare training data
trainX = encode_sequences(train_tokenizer, input_length, train[:, 1])
trainY = encode_sequences(train_tokenizer, input_length, train[:, 2])
trainY = encode_output(trainY, train_vocab_size)
# prepare test data
testX = encode_sequences(train_tokenizer, input_length, test[:, 1])
testY = encode_sequences(train_tokenizer, input_length, test[:, 2])
testY = encode_output(testY, train_vocab_size)

# define model
model = define_model(train_vocab_size, dim_size, input_length, units, embedding_weights)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize defined model
print(model.summary())
# fit model
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
