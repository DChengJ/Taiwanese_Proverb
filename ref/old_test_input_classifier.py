import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.callbacks import TensorBoard
import numpy as np
import re
import os
import jieba

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './res', 'path to directory')
tf.app.flags.DEFINE_string('Train_set', '0m_TR.txt', '')
tf.app.flags.DEFINE_string('Test_set', '0m_TE.txt', '')
tf.app.flags.DEFINE_integer('limit_v', 1, 'Word Vector Frequence > 1')
tf.app.flags.DEFINE_integer('num_classes', 3, 'Number of Labels')
tf.app.flags.DEFINE_boolean('remove_mid', False, 'Remove Label Mid')

dropout_rate = 0.7

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

def LoadHeader(path):
    header=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')[0]
            header.append(line)
    return header

def calulateWordVec(sentence, header):
    new_data = []
    sentence_wordvector = [0] * len(header)
    for index, word in enumerate(header):
        exist_word = False
        for v in sentence:
            if word == v:
                sentence_wordvector[index] +=1
    new_data.append(sentence_wordvector)
    return np.array(new_data, np.int32)

def build_model(nb_input_vector, num_classes):
    model = Sequential()
    model.add(Dense(1024, activation='tanh', input_dim=nb_input_vector))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(dropout_rate))
    #model.add(LSTM(units=nb_lstm_outputs, input_shape= (nb_time_steps, nb_input_vector), return_sequences=True))
    #model.add(Activation('tanh'))
    #model.add(Dropout(dropout_rate))
    #model.add(LSTM(nb_lstm_outputs, return_sequences=True))
    #model.add(Dropout(dropout_rate))
    #model.add(LSTM(nb_lstm_outputs))
    #model.add(Activation('tanh'))
    #model.add(Dropout(dropout_rate))    
    #model.add(Activation(activation))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def main(argv):
    assert FLAGS.data_path and FLAGS.num_classes
    data_path = FLAGS.data_path
    num_classes = FLAGS.num_classes
    epochs_step = 20
    _batch_size = 100
    wordvector_path = '%s/0m_HE.txt' % (data_path)
    model_header = LoadHeader(wordvector_path)
    while True:
        input_sentence = input('Input a sentence:')
        if input_sentence == '0' or input_sentence == '': break
        sentence = segment(input_sentence)
        x_test = calulateWordVec(sentence, model_header)
        nb_input_vector = len(x_test)
        print("Word Vector Len: " + str(nb_input_vector))
        save_path = "%s/%s" % (data_path, "m_MODEL.h5")
        print(save_path)
        if not os.path.isfile(save_path):
            print('Not Model File!!')
            callback = TensorBoard(log_dir='./model', histogram_freq=0,  write_graph=True, write_images=True)
            model = build_model(nb_input_vector, num_classes)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Start Train")
            model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, verbose=2)
            model.save(save_path)
        else:
            print('Yes')
            model = tf.contrib.keras.models.load_model(save_path)
        print("End Train")
        pred = model.predict(x_test, batch_size=_batch_size, verbose=0)
        print(pred)
        pred = np.argmax(pred, 1)
        print(pred+1)
        
if __name__ == '__main__':
    tf.app.run()
