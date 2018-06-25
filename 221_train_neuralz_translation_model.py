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
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
np.random.seed(1)

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('num_CV', 5, 'N-cross-validation')
tf.app.flags.DEFINE_string('word2vec_path', 'word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt', 'path to directory')
tf.app.flags.DEFINE_string('dataset_path', 'crossvalidation/2english-german-both.pkl', 'all dataset of path to directory')
tf.app.flags.DEFINE_string('train_path', 'crossvalidation/2english-german-test.pkl', 'train dataset of path to directory')
tf.app.flags.DEFINE_string('test_path', 'crossvalidation/2english-german-train.pkl', 'test dataset of path to directory')
tf.app.flags.DEFINE_integer('input_length', 15, 'Input sentence length')
tf.app.flags.DEFINE_integer('dim_size', 512, 'The Dimensions of Word2Vec')
tf.app.flags.DEFINE_integer('units', 512, 'Neural network units')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('epochs', 20, 'Epochs size')

# load dataset
def load_sentences(data_path):
    return load(open(data_path, 'rb'))

# load Word2Vec
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
def define_model(X_vocab_size, Y_vocab_size, X_length, Y_length, units):
    model = Sequential()
    model.add(Embedding(input_dim=X_vocab_size, output_dim=units, input_length=X_length, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(Y_length))
    model.add(LSTM(units, return_sequences=True))
    model.add(TimeDistributed(Dense(Y_vocab_size, activation='softmax')))
    return model

def main(argv):
    # parameter
    wv_path = FLAGS.word2vec_path
    input_length = FLAGS.input_length
    dim_size = FLAGS.dim_size
    units = FLAGS.units
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    model_path = 'models/seq2seq/modelv2.h5'

    #dataset parameter
    dataset_path = FLAGS.dataset_path
    train_path = FLAGS.train_path
    test_path = FLAGS.test_path

    # load datasets
    dataset = load_sentences(dataset_path)
    train = load_sentences(train_path)
    test = load_sentences(test_path)
    
    X_tokenizer = create_tokenizer(dataset[:, 1])
    X_vocab_size = len(X_tokenizer.word_index) + 1
    X_length = max_length(dataset[:, 1])
    print('X Vocabulary Size: %d' % X_vocab_size)
    print('X Max Length: %d' % (X_length))

    Y_tokenizer = create_tokenizer(dataset[:, 0])
    Y_vocab_size = len(Y_tokenizer.word_index) + 1
    Y_length = max_length(dataset[:, 0])
    print('Y Vocabulary Size: %d' % Y_vocab_size)
    print('Y Max Length: %d' % (Y_length))

    # prepare training data
    trainX = encode_sequences(X_tokenizer, X_length, train[:, 1])
    trainY = encode_sequences(Y_tokenizer, Y_length, train[:, 0])
    trainY = encode_output(trainY, Y_vocab_size)
    # prepare test data
    testX = encode_sequences(X_tokenizer, X_length, test[:, 1])
    testY = encode_sequences(Y_tokenizer, Y_length, test[:, 0])
    testY = encode_output(testY, Y_vocab_size)

    print('Start')
    # define model
    modelT_s = time.time()
    model = define_model(X_vocab_size, Y_vocab_size, X_length, Y_length, units)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize defined model
    print(model.summary())
    # fit model
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
    modelT_e = time.time()
    print('Train Model elapsed time: %.2f sec.' % (modelT_e - modelT_s))


if __name__ == '__main__':
    tf.app.run()