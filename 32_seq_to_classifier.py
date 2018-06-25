import numpy as np
import tensorflow as tf
import keras
import pickle
import re
import os
import time
from pickle import load
from numpy import array
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.callbacks import TensorBoard
from keras.models import load_model
np.random.seed(1)

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('num_CV', 1, 'N-cross-validation')
tf.app.flags.DEFINE_string('word2vec_path', 'word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt', 'path to directory')
tf.app.flags.DEFINE_string('dataset_path', 'crossvalidation/newProverb2_jieba_v2_1-both.pkl', 'all dataset of path to directory')
tf.app.flags.DEFINE_string('train_path', 'crossvalidation/newProverb2_jieba_v2_1-1-train.pkl', 'train dataset of path to directory')
tf.app.flags.DEFINE_string('test_path', 'crossvalidation/newProverb2_jieba_v2_1-1-test.pkl', 'test dataset of path to directory')
tf.app.flags.DEFINE_string('model_path', 'model.h5', 'test dataset of path to directory')
tf.app.flags.DEFINE_string('typeA', 'Question', 'Choice Analysis tpye')
tf.app.flags.DEFINE_integer('input_length', 10, 'Input sentence length')
tf.app.flags.DEFINE_integer('dim_size', 512, 'The Dimensions of Word2Vec')
tf.app.flags.DEFINE_integer('units', 512, 'Neural network units')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('epochs', 10, 'Epochs size')
tf.app.flags.DEFINE_bool('TrainC', True, 'Restart')


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

def process_sentences(data, num_classes, typeA):
    labels = list()
    labels_onehot = list()
    sentences = list()
    for line in data:
        label = int(line[0])
        onehot = list()
        for i in range(num_classes):
            if i == (label - 1):
                onehot.append(1)
            else:
                onehot.append(0)
        labels_onehot.append(onehot)
        labels.append(label)
        if typeA == 'All':
            #print('All')
            sentence = line[1] + ' ' + line[2]
        elif typeA == 'Question':
            #print('Question')
            sentence = line[1]
        elif typeA == 'Answer':
            #print('Answer')
            sentence = line[2]
        sentences.append(sentence)
    return np.array(sentences), np.array(labels_onehot), np.array(labels)

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

def train_lstm_model(vocab, dim_size, input_length, embedding_weights, units, class_num):
    model = Sequential()
    model.add(Embedding(input_dim=vocab,
                        output_dim=dim_size,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(Dropout(0.7))
    model.add(LSTM(units, activation='relu', return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(units, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(3, activation='softmax'))
    return model

def main(argv):
    # parameter
    wv_path = FLAGS.word2vec_path
    input_length = FLAGS.input_length
    dim_size = FLAGS.dim_size
    units = FLAGS.units
    _batch_size = FLAGS.batch_size
    epochs_step = FLAGS.epochs
    typeA = FLAGS.typeA
    SaveModel_path = 'models/classifier/' + FLAGS.model_path
    index_dict = None
    word_vectors = None
    class_num = 3
    TrainC = FLAGS.TrainC
    seed = 1

    #dataset parameter
    dataset_path = FLAGS.dataset_path
    train_path = FLAGS.train_path
    test_path = FLAGS.test_path

    # Load datasets
    print('Loading Dataset ...')
    dataset = load_sentences(dataset_path)
    train = load_sentences(train_path)
    test = load_sentences(test_path)
    print('Finished Load')

    # load Word2Vec
    print('Loading Word2Vec ...')
    loadWV_s = time.time()
    # words_index, words_vectors = load_word2vec_pkl(pkl_path)
    words_index, words_vectors = load_word2vec(wv_path)
    loadWV_e = time.time()
    print('Finished Load elapsed time: %.2f sec.' % (loadWV_e - loadWV_s))

    # prepare train tokenizer
    sentences, words_vectors = compare_WV(dataset, words_vectors)
    #texts = [word for word in words_index]

    tokenizer = create_tokenizer(sentences)
    train_vocab_size = len(tokenizer.word_index) + 1
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Process Sentences
    trainX, trainY, trainL = process_sentences(train, class_num, typeA)
    testX, testY, testL = process_sentences(test, class_num, typeA)
    X , Y, labels = process_sentences(dataset, class_num, typeA)
    length = max_length(X)
    print("資料長度：", length)
    print("資料集筆數：", len(dataset))

    # Calculated Embedding Weights
    embedding_weights = calculation_embedding_weights(words_vectors, tokenizer, train_vocab_size, dim_size)
    del words_vectors
    print("詞權重數量", len(embedding_weights), ", 字詞維度", len(embedding_weights[0]))
    trainX = encode_sequences(tokenizer, input_length, trainX[:])
    testX = encode_sequences(tokenizer, input_length, testX[:])
    X = encode_sequences(tokenizer, input_length, X[:])

    cvscores = list()

    model = None
    for train, test in kfold.split(X, labels):
        if not os.path.isfile(SaveModel_path) or TrainC:
            #print('Not Exist Model')
            TrainModel_s = time.time()
            #callback = TensorBoard(log_dir='log_root', histogram_freq=0,  write_graph=True, write_images=True)
            print('Setup Model')
            model  = train_lstm_model(train_vocab_size, dim_size, input_length, embedding_weights, units, class_num)
            print('Compile Model')
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Start Train")
            #model.fit(trainX, trainY, epochs=epochs_step, batch_size=_batch_size, callbacks=[callback], verbose=1)
            model.fit(X[train], Y[train], epochs=epochs_step, batch_size=_batch_size, verbose=0)
            TrainModel_e = time.time()
            #model.save(SaveModel_path)
            print('Train Classifier Model Finished elapsed time: %.2f sec.' % (TrainModel_e - TrainModel_s))
        else:
            #print('Exist Model')
            model = tf.contrib.keras.models.load_model(SaveModel_path)

        pred = model.predict(X[test], batch_size=_batch_size, verbose=0)
        pred = np.argmax(pred, 1).tolist()
        truth = np.argmax(Y[test], 1).tolist()
        precision, recall, f1 = self_metric(truth, pred)

        print("Start Evaluate")
        #score, accuracy = model.evaluate(X[train], Y[train],batch_size=_batch_size, verbose=0)
        #print('Train score:', score)
        #print('Train accuracy:', accuracy)
        #score, accuracy = model.evaluate(X[test], Y[test],batch_size=_batch_size, verbose=0)
        #print('Test score:', score)
        #print('Test accuracy:', accuracy)

        scores = model.evaluate(X[test], Y[test], verbose=0)
        cvscores.append(scores[1] * 100)

        print('F1值', f1)
        print("%.1f%%(%.2f)" % (round(scores[1]*10000)/100, f1))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == '__main__':
    tf.app.run()