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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
np.random.seed(1)

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('num_CV', 5, 'N-cross-validation')
tf.app.flags.DEFINE_string('word2vec_path', 'word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt', 'path to directory')
tf.app.flags.DEFINE_string('dataset_path', 'crossvalidation/2english-german-both.pkl', 'all dataset of path to directory')
tf.app.flags.DEFINE_string('train_path', 'crossvalidation/2english-german-test.pkl', 'train dataset of path to directory')
tf.app.flags.DEFINE_string('test_path', 'crossvalidation/2english-german-train.pkl', 'test dataset of path to directory')
tf.app.flags.DEFINE_integer('input_length', 15, 'Input sentence length')
tf.app.flags.DEFINE_integer('dim_size', 512, 'The Dimensions of Word2Vec')


# load a clean dataset
def load_sentences(filename):
        return load(open(filename, 'rb'))

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

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    print(len(source[0]))
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    print('預測標號值', integers)
    target = list()
    for i in integers:
            word = word_for_id(i, tokenizer)
            if word is None:
                    break
            target.append(word)
    return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, X_tokenizer, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        print('第', i+1, '個')
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_src, raw_target = raw_dataset[i]
        words = list()
        for s in source[0]:
            words.append(word_for_id(s, X_tokenizer))
        print('謎面編號：', source)
        print('謎面：利用編號再轉換成文字：', words)
        print('謎面：', raw_src)
        print('謎底：', raw_target)
        print('預測謎底：', translation)
        print('i=[%s], src=[%s], target=[%s], predicted=[%s]' % (i, raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def main(argv):
    # parameter
    # parameter
    wv_path = FLAGS.word2vec_path
    input_length = FLAGS.input_length
    dim_size = FLAGS.dim_size
    model_path = 'models/seq2seq/modelv2.h5'

    #dataset parameter
    dataset_path = FLAGS.dataset_path
    train_path = FLAGS.train_path
    test_path = FLAGS.test_path

    # Load datasets
    dataset = load_sentences(dataset_path)
    train = load_sentences(train_path)
    test = load_sentences(test_path)
    # prepare X tokenizer
    X_tokenizer = create_tokenizer(dataset[:, 1])
    X_vocab_size = len(X_tokenizer.word_index) + 1
    X_length = max_length(dataset[:, 1])
    print('X Vocabulary Size: %d' % X_vocab_size)
    print('X Max Length: %d' % (X_length))
    # prepare Y tokenizer
    Y_tokenizer = create_tokenizer(dataset[:, 0])
    Y_vocab_size = len(Y_tokenizer.word_index) + 1
    Y_length = max_length(dataset[:, 0])
    print('Y Vocabulary Size: %d' % Y_vocab_size)
    print('Y Max Length: %d' % (Y_length))

    # prepare data
    trainX = encode_sequences(X_tokenizer, X_length, train[:, 1])
    testX = encode_sequences(X_tokenizer, X_length, test[:, 1])

    # load model
    model = load_model(model_path)
    # test on some training sequences
    print('train')
    evaluate_model(model, X_tokenizer, Y_tokenizer, trainX, train)

    # test on some test sequences
    print('test')
    evaluate_model(model, X_tokenizer, Y_tokenizer, testX, test)

if __name__ == '__main__':
    tf.app.run()