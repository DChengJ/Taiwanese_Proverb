import pickle
import re
import os
import numpy as np
np.random.seed(1)

import tensorflow as tf
import keras
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence

file_path = 'newProverb2_jieba_v1_1.csv'
pkl_path = 'Word2Vec_Dict.pkl'

def LoadData(path):
    line_list = []
    label_list = []
    maxV = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            label, text1, text2 = list(filter(None, re.split(r',+|\"+', line)))
            
            # Text
            text = text1+ ' ' + text2
            sen_list = list(filter(None, re.split(' ', text)))
            if maxV < len(sen_list): maxV = len(sen_list)
            line_list.append(sen_list)
            # Label
            tmplist = []
            label = int(label)
            for i in range(3):
                if i == (label - 1):
                    tmplist.append(1)
                else:
                    tmplist.append(0)
            label_list.append(tmplist)
    return label_list, line_list

def text_to_word2vec(pkl_dict, sentences):
    new_sentences = []
    for sen in sentences:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(pkl_dict[word])
            except:
                new_sen.append(0)
        new_sentences.append(new_sen)
    return np.array(new_sentences)

def train_lstm_model(top_words, vocab_dim, input_length, embedding_weights, x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(input_dim=top_words,
                        output_dim=vocab_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dropout(0.7))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

# initial parameter
index_dict = None
word_vectors = None
maxlen = 30
totel_words = None
top_words = 20000
vocab_dim = 300
input_length = 30
SaveModel_path = './model.h5'
epochs_step = 30
_batch_size = 20

print('Loading Dataset ...')
labels, sentences = LoadData(file_path)
print('Finished Load')

# Load NLP Data
print('Loading Word2Vec and Dict ...')
with open(pkl_path, 'rb') as f:
    index_dict = pickle.load(f)
    word_vectors = pickle.load(f)
print('Finished Load')

# Calculated quantity 
totel_words = len(index_dict) + 1
embedding_weights = np.zeros((totel_words, 300))
for w, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[w]

x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

x_train, x_test = text_to_word2vec(index_dict, x_train), text_to_word2vec(index_dict, x_test)
y_train, y_test = np.array( y_train), np.array( y_test)
print('train dataset shape: ', x_train.shape)
print('test dataset shape: ', x_test.shape)

print('Padding Sequences (samples time steps)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('After Padding')
print('train dataset shape: ', x_train.shape)
print('test dataset shape: ', x_test.shape)

model = None
if not os.path.isfile(SaveModel_path):
    print('Not Exist Model')
    callback = TensorBoard(log_dir='./model', histogram_freq=0,  write_graph=True, write_images=True)
    print('Setup Model')
    model  = train_lstm_model(totel_words, vocab_dim, input_length, embedding_weights, x_train, y_train, x_test, y_test)
    print('Compile Model')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Start Train")
    model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, callbacks=[callback], verbose=2)
    #model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, verbose=2)
    model.save(SaveModel_path)
else:
    print('Exist Model')
    model = tf.contrib.keras.models.load_model(SaveModel_path)

score, accuracy = model.evaluate(x_test, y_test,batch_size=_batch_size, verbose=1)
print('Test score:', score)
print('Test accuracy:', accuracy)

