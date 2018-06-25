import tensorflow as tf
import time
from gensim.models import word2vec

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('dim_size', 100, 'Dimensionality of the feature vectors.')
tf.app.flags.DEFINE_integer('sg', 1, '1: skip-gram, 0: CBOW')
tf.app.flags.DEFINE_integer('workers', 4, ' Use these many worker threads to train the model')
tf.app.flags.DEFINE_integer('min_count', 5, 'Ignores all words with total frequency lower than this.')
tf.app.flags.DEFINE_integer('window', 3, 'The maximum distance between the current and predicted word within a sentence.')
tf.app.flags.DEFINE_string('sentences_path', './word2vec_data/segmentation.txt', 'docs path to directory')
tf.app.flags.DEFINE_string('fvocab_path', './word2vec_data/vocab.txt', 'docs path to directory')
tf.app.flags.DEFINE_string('output_path', './word2vec', 'Word2Vec Bin File output path to directory')
tf.app.flags.DEFINE_string('filename_path', '', 'Word2Vec Bin File output path to directory')

def proverb(data_path):
    print()

def main(argv):   
    sentences_path = FLAGS.sentences_path
    fvocab = FLAGS.fvocab_path
    output_path = FLAGS.output_path
    dim_size = FLAGS.dim_size
    sg = FLAGS.sg
    workers = FLAGS.workers
    min_count = FLAGS.min_count
    window = FLAGS.window
    filename_path = FLAGS.filename_path

    print("訓練中...(喝個咖啡吧^0^)")
    ts = time.time()
    sentence = word2vec.Text8Corpus(sentences_path)
    model = word2vec.Word2Vec(sentence, size = dim_size, window=window, min_count=min_count, workers=workers, sg=sg)
    output_path = "%s/%s%s.model.bin" % (output_path, filename_path, dim_size)
    model.wv.save_word2vec_format(output_path, fvocab=fvocab, binary = True)
    te = time.time()
    print("model 已儲存完畢: %.2f sec" % (te-ts))
    
if __name__ == '__main__':
    tf.app.run()
