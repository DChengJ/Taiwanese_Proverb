import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora.dictionary import Dictionary
import re
import warnings
from pickle import dump
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('wv_path', 'word2vec/wiki300.model.bin', 'path to directory')
tf.app.flags.DEFINE_string('save_path', 'word2vec/', 'path to directory')

def create_dictionaries(word2vec):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(word2vec.vocab.keys(), allow_update=True)
    word_index = {word: index + 1 for index, word in gensim_dict.items()}
    word_vector = {word: word2vec[word] for word in word_index.keys()}
    return word_index, word_vector

def main(_):
    wv_path = FLAGS.wv_path
    save_path = FLAGS.save_path
    dim_size = re.compile("[^0-9]").sub("", wv_path.rsplit('/', 1)[-1].split('.', 1)[0])
    filename = dim_size + 'Word2Vec_Dict.pkl'
    save_path += filename
    
    print('Load Word2Vec')
    word2vectors = KeyedVectors.load_word2vec_format(wv_path, binary = True)
    print('Load Finished')
    # words 616839
    # docs2 = ' '.join(word2vectors.vocab.keys())
    
    print('Analysis Words Vocab')
    words_dict, word_vectors = create_dictionaries(word2vectors)
    '''
    with open('dict2.txt', 'w', encoding='utf-8') as f:
        for word in words_dict:
            f.write(word + '\n')
    '''
    print('Save Word2Vec and Word2Token')    
    with open(save_path, 'wb') as f:
        dump(words_dict, f)
        dump(word_vectors, f)
    print('Saved')
    
if __name__ == '__main__':
    tf.app.run()
