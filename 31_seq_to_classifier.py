import numpy as np
import tensorflow as tf
import keras
import pickle
import re
import os
np.random.seed(1)

from pickle import load
from numpy import array

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.callbacks import TensorBoard
from keras.models import load_model

# load a clean dataset
def load_sentences(filename):
        return load(open(filename, 'rb'))

# load Word2Vec
def load_word2vec_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        words_index = load(f)
        words_vectors = load(f)
    return words_index, words_vectors

def load_word2vec(wv_path):
    words_vectors = {}
    words = []
    with open(wv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            word = line.split('\t')[0]
            wv = re.split(r'\t', line, maxsplit=1)[1].split('\t')
            words.append(word)
            words_vectors[word] = [float(v) for v in wv]
    return words, words_vectors

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
    return lines, new_wv

def calculation_embedding_weights(words_vectors, tokenizer, vocab_size, dim_size):
    embedding_weights = np.zeros((vocab_size, dim_size))
    for word, vector in words_vectors.items():
        index = tokenizer.texts_to_sequences([word])
        embedding_weights[index, :] = vector
    return embedding_weights

def process_sentences(data, num_classes):
    labels = []
    sentences = []
    for i in range(len(data)):
    	label = int(data[i, 0])
    	tmplist = []
    	for i in range(num_classes):
    		if i == (label-1):
    			tmplist.append(1)
    		else:
    			tmplist.append(0)
    	labels.append(tmplist)
    	sentence = data[i, 1] + " " + data[i, 2]
    	sentences.append(sentence)
    return np.array(sentences), np.array(labels)

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

# Confusion Matrix
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

def self_metric(truth_y, pred_y):
	print("Metrics")
	sess = tf.Session()
	print(sess.run(tf.contrib.metrics.confusion_matrix(labels=truth_y, predictions=pred_y)))
	precision, recall, f1 = self_confusionM(truth_y, pred_y)
	return precision, recall, f1 

'''
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
'''

def train_lstm_model(top_words, vocab_dim, input_length, embedding_weights, units, class_num):
    model = Sequential()
    model.add(Embedding(input_dim=top_words,
                        output_dim=vocab_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(units, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(units, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(units, activation='sigmoid'))
    model.add(Dropout(0.7))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    return model

# File
data_path = 'newProverb2_jieba_v4_0.csv'
pkl_path = 'word2vec/300Word2Vec_Dict.pkl'
wv_path = 'word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt'

# initial parameter
dim_size = 512
index_dict = None
word_vectors = None
input_length = 20
units = 512
SaveModel_path = 'models/classifier/model.h5'
epochs_step = 30
_batch_size = 20
class_num = 3

# Load datasets
print('Loading Dataset ...')
dataset = load_sentences('./crossvalidation/newProverb2_jieba_v2_0-both.pkl')
train = load_sentences('./crossvalidation/newProverb2_jieba_v2_0-train.pkl')
test = load_sentences('./crossvalidation/newProverb2_jieba_v2_0-test.pkl')
print('Finished Load')

# Load Word2Vec
# print('Loading Word2Vec and Dict ...')
# words_index, words_vectors = load_word2vec_pkl(pkl_path)
# print('Finished Load')
print('Loading Word2Vec and Dict ...')
words_index, words_vectors = load_word2vec(wv_path)
print('Finished Load')

# prepare train tokenizer
token_train, words_vectors = compare_WV(train, words_vectors)

'''
texts = []
for word in words_index:
    texts.append([word])
'''    

train_tokenizer = create_tokenizer(token_train)
train_vocab_size = len(train_tokenizer.word_index) + 1

# Process Sentences
trainX, trainY = process_sentences(train, class_num)
testX, testY = process_sentences(test, class_num)
X_length = max_length(trainX)
Y_length = max_length(testX)
print(X_length)
print(Y_length)

# Calculated Embedding Weights
embedding_weights = calculation_embedding_weights(words_vectors, train_tokenizer, train_vocab_size, dim_size)
trainX = encode_sequences(train_tokenizer, input_length, trainX[:])
testX = encode_sequences(train_tokenizer, input_length, testX[:])

# prepare training data

'''
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
'''
del words_vectors
model = None
if not os.path.isfile(SaveModel_path):
    print('Not Exist Model')
    callback = TensorBoard(log_dir='./log_root', histogram_freq=0,  write_graph=True, write_images=True)
    print('Setup Model')
    model  = train_lstm_model(train_vocab_size, dim_size, input_length, embedding_weights, units, class_num)
    print('Compile Model')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Start Train")
    model.fit(trainX, trainY, epochs=epochs_step, batch_size=_batch_size, callbacks=[callback], verbose=1)
    #model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, verbose=2)
    model.save(SaveModel_path)
else:
    print('Exist Model')
    model = tf.contrib.keras.models.load_model(SaveModel_path)

pred = model.predict(testX, batch_size=_batch_size, verbose=0)
pred = np.argmax(pred, 1).tolist()
truth = np.argmax(testY, 1).tolist()
precision, recall, f1 = self_metric(truth, pred)

print("Start Evaluate")
score, accuracy = model.evaluate(testX, testY,batch_size=_batch_size, verbose=1)
print('Test score:', score)
print('Test accuracy:', accuracy)
