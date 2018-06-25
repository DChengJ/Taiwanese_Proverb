import numpy as np
import tensorflow as tf
import keras
import pickle
import re
import os
import time
from pickle import load
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.contrib.keras.api.keras.initializers import Constant
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
np.random.seed(1)

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('num_CV', 5, 'N-cross-validation')
tf.app.flags.DEFINE_string('word2vec_path', 'word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt', 'path to directory')
tf.app.flags.DEFINE_string('dataset_path', 'crossvalidation/newProverb2_jieba_v3_1-both.pkl', 'all dataset of path to directory')
tf.app.flags.DEFINE_string('train_path', 'crossvalidation/newProverb2_jieba_v3_1-1-train.pkl', 'train dataset of path to directory')
tf.app.flags.DEFINE_string('test_path', 'crossvalidation/newProverb2_jieba_v3_1-1-test.pkl', 'test dataset of path to directory')
tf.app.flags.DEFINE_string('model_path', 'model.h5', 'model of path to directory')
tf.app.flags.DEFINE_integer('input_length', 15, 'Input sentence length')
tf.app.flags.DEFINE_integer('dim_size', 512, 'The Dimensions of Word2Vec')
tf.app.flags.DEFINE_integer('units', 512, 'Neural network units')
tf.app.flags.DEFINE_integer('batch_size', 200, 'Batch size')
tf.app.flags.DEFINE_integer('epochs', 5, 'Epochs size')

# load dataset
def load_sentences(data_path):
    return load(open(data_path, 'rb'))

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
    model.add(Embedding(input_dim=vocab, output_dim=dim_size, input_length=timesteps, weights=[embedding_weights], mask_zero=True)) #, embeddings_initializer=Constant(embedding_weights)
    model.add(LSTM(n_units))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab, activation='softmax')))
    return model

def main(argv):
    # parameter
    wv_path = FLAGS.word2vec_path
    input_length = FLAGS.input_length
    dim_size = FLAGS.dim_size
    units = FLAGS.units
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    model_path = 'models/seq2seq/'+ FLAGS.model_path

    #dataset parameter
    dataset_path = FLAGS.dataset_path
    train_path = FLAGS.train_path
    test_path = FLAGS.test_path

    # load datasets
    dataset = load_sentences(dataset_path)
    train = load_sentences(train_path)
    test = load_sentences(test_path)

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
    train_tokenizer = create_tokenizer(sentences)
    print('Tokenizer Finished')
    train_vocab_size = len(train_tokenizer.word_index) + 1
    X_length = max_length(dataset[:, 1])
    Y_length = max_length(dataset[:, 2])
    print("資料集筆數：", len(dataset))
    print("謎面最大長度：", X_length)
    print("謎底最大長度：", Y_length)

    # The output word vector is after the dataset and the word vector
    #with open('train1_0.txt', 'w', encoding='utf-8') as f:
    #	for word, value in train_tokenizer.word_index.items():
    #		f.write("%s\t%s\n" % (word, value))
    
    # prepare embedding weight
    embedding_weights = calculation_embedding_weights(words_vectors, train_tokenizer, train_vocab_size, dim_size)
    print(embedding_weights.shape)
    print("詞權重數量", len(embedding_weights), ", 字詞維度", len(embedding_weights[0]))
    del words_vectors

    # prepare training data
    trainX = encode_sequences(train_tokenizer, input_length, train[:, 1])
    trainY = encode_sequences(train_tokenizer, input_length, train[:, 2])
    trainY = encode_output(trainY, train_vocab_size)
    # prepare test data
    testX = encode_sequences(train_tokenizer, input_length, test[:, 1])
    testY = encode_sequences(train_tokenizer, input_length, test[:, 2])
    testY = encode_output(testY, train_vocab_size)
    print("字詞轉換長度", len(testY[0][0]))
    print("句子字詞數", len(testY[0]))
    print('Start')
    # define model
    modelT_s = time.time()
    for word, index in train_tokenizer.word_index.items():
        if index == 1:
            print(word)
    model = define_model(train_vocab_size, dim_size, input_length, units, embedding_weights)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize defined model
    print(model.summary())
    # fit model
    #checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
    modelT_e = time.time()
    print('Train Model elapsed time: %.2f sec.' % (modelT_e - modelT_s))

if __name__ == '__main__':
    tf.app.run()