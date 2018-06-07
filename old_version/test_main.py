import pickle
import re
import os
import numpy as np
np.random.seed(1)

import tensorflow as tf
import keras
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './data', 'path to directory')
tf.app.flags.DEFINE_string('file_path', './data/newProverb2_jieba_v1_1.csv', 'file path to directory')
tf.app.flags.DEFINE_string('word2vec_path', './word2vec/Word2Vec_Dict.pkl', 'word2vec path to directory')
tf.app.flags.DEFINE_string('model_path', './models/classifier/model.h5', 'classifier model path to directory')
tf.app.flags.DEFINE_integer('epochs_step', 20, 'epochs_step')
tf.app.flags.DEFINE_integer('_batch_size', 20, '_batch_size')
tf.app.flags.DEFINE_integer('num_classes', 3, 'Number of Labels')

def self_confusionM(cor_list, pred_list):
    class_list = []
    cor_class = {}
    pred_class = {}
    True_class = {}
    totals = 0
    all_precision = all_recall = all_f1 = 0
    
    for index, cor_v in enumerate(cor_list):
        pred_v = pred_list[index]
        if cor_v == pred_v:
            if not cor_v in True_class:
                True_class[cor_v] = 1
            else:
                num = True_class[cor_v]
                True_class[cor_v] = (num + 1)
        # Calculation cor_list
        if not cor_v in cor_class:
            cor_class[cor_v] = 1
        else:
            num = cor_class[cor_v]
            cor_class[cor_v] = (num + 1)
        # Calculation pred_list
        if not pred_v in pred_class:
            pred_class[pred_v] = 1
        else:
            num = pred_class[pred_v]
            pred_class[pred_v] = (num + 1)
        # Calculation class_list 
        if not cor_v  in class_list:
            class_list.append(cor_v)
    # Check pred_class contain class_list all value
    for class_i in class_list:
        totals += cor_class[class_i]
        if not class_i in True_class:
            True_class[class_i] = 0
        if not class_i in pred_class:
            pred_class[class_i] = 0
    print('Multi Class Confusion Matrix')
    print('Class\tPrecision\tRecall\t\tF-Measure\tNumbers')
    for class_i in class_list:
        pc = float(pred_class[class_i])
        cc = float(cor_class[class_i])
        tc = float(True_class[class_i])
        if pc == 0:
            precision = 0
        else:
            precision = tc/pc
        if cc ==0:
            recall = 0
        else:
            recall = tc / cc
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        all_precision +=precision * cor_class[class_i] / totals
        all_recall += recall * cor_class[class_i] / totals
        all_f1 += f1 * cor_class[class_i] / totals
        print('%d\t%.3f\t\t%.3f\t\t%.3f\t\t%d' % (class_i, precision, recall, f1, cor_class[class_i]))
    print('---------------------------------------------------')
    print('%s\t%.3f\t\t%.3f\t\t%.3f\t\t%d' % ('all', all_precision, all_recall, all_f1, totals))
    return all_precision, all_recall, all_f1

def LoadData(path):
    line_list = []
    label_list = []
    maxV = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            label, text1, text2 = re.split(r',\"|\",\"|\"', line[:-1])
            # Text
            text = text1+ ' ' + text2
            sen_list = re.split(' ', text)
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

def train_lstm_model(vocab_words, vocab_dim, input_length, embedding_weights, x_train, y_train, x_test, y_test):
    print(vocab_words, vocab_dim, input_length)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_words,
                        output_dim=vocab_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
    #model.add(LSTM(100, activation='sigmoid', return_sequences=True))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    return model

def self_metric(model, x, y):
    pred = model.predict(x, batch_size=30 ,verbose=0)
    sess = tf.Session()
    truth_y = np.argmax(y, 1).tolist()
    pred_y = np.argmax(pred, 1).tolist()
    print(sess.run(tf.contrib.metrics.confusion_matrix(labels=truth_y, predictions=pred_y)))
    precision, recall, f1 = self_confusionM(truth_y, pred_y)
    print("F1: %.4f" % (f1))


# Setup parameters
num_classes = None
vocab_dim = 300
input_length = 30

def main(argv):
    file_path = FLAGS.file_path
    model_path = FLAGS.model_path
    word2vec_path = FLAGS.word2vec_path
    global num_classes
    num_classes = FLAGS.num_classes
    epochs_step = FLAGS.epochs_step
    _batch_size = FLAGS._batch_size
    
    print('Loading Dataset ...')
    labels, sentences = LoadData(file_path)
    print('Finished Load')
    # Load NLP Data
    print('Loading Word2Vec and Dict ...')
    with open(word2vec_path, 'rb') as f:
        index_dict = pickle.load(f)
        word_vectors = pickle.load(f)
    print('Finished Load')
    # Calculated quantity
    vocab_len = len(index_dict) + 1
    embedding_weights = np.zeros((vocab_len, 300))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
        
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)
    x_train, x_test = text_to_word2vec(index_dict, x_train), text_to_word2vec(index_dict, x_test)
    y_train, y_test = np.array( y_train), np.array( y_test)
    print('train dataset shape: ', x_train.shape)
    print('test dataset shape: ', x_test.shape)

    print('Padding Sequences (samples time steps)')
    #x_train = sequence.pad_sequences(x_train, maxlen=input_length, padding='post')
    x_train = pad_sequences(x_train, maxlen=input_length)
    x_test = pad_sequences(x_test, maxlen=input_length)
    print('After Padding')
    print('train dataset shape: ', x_train.shape)
    print('test dataset shape: ', x_test.shape)
    print(x_train[0])
    t = list(index_dict.keys())
    print(t[691202])
    print(t[505942])
    print(t[206514])
    print(t[696102])
    print(t[696068])
    # del var
    del index_dict
    del word_vectors
    
    if not os.path.isfile(model_path):
        print('Not Exist Model')
        #callback = TensorBoard(log_dir='./log_root', histogram_freq=0,  write_graph=True, write_images=True)
        print('Initialization Model')
        model  = train_lstm_model(vocab_len, vocab_dim, input_length, embedding_weights, x_train, y_train, x_test, y_test)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, verbose=1)
        model.save(model_path)
    else:
        print('Exist Model')
        model = load_model(model_path)
        
    print("Start Evaluate")
    self_metric(model, x_test, y_test)
    result = model.evaluate(x_train, y_train,batch_size=_batch_size, verbose=1)
    print ("Train accuracy: %.4f" % (result[1]))
    result = model.evaluate(x_test, y_test,batch_size=_batch_size, verbose=1)
    print ("Test accuracy: %.4f" % (result[1]))
    
if __name__ == '__main__':
    tf.app.run()
