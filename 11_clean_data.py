import tensorflow as tf
import numpy as np
import collections
import string
#import tools.jiebazhtw as jieba
import jieba
import re
from pickle import dump
from hanziconv import HanziConv

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './data/newProverb2_v2_1.csv', 'path to directory')
tf.app.flags.DEFINE_string('dict_path', None, 'dict path to directory')
tf.app.flags.DEFINE_string('output_path', './pkl_data/newProverb2_jieba_v2_1.pkl', 'output path to directory')


def load_data(data_path):
    datas= None
    with open(data_path, 'rt', encoding='utf-8') as f:
        datas = f.read()
    return datas

def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [re.split(r'\",\"', line[1:-1]) for line in  lines]
    return pairs

def clean_pair(lines):
    cleaned = []
    rule = re.compile(r'[^\u4e00-\u9fa5]')
    for pair in lines:
        if len(pair) != 3:
            print(pair)
            continue
        clean_pair = []
        for index, line in enumerate(pair):
            if index == 0:
                clean_pair.append(line)
                continue
            line = rule.sub("", line)
            #line = HanziConv.toSimplified(line)
            # line = [' '.join(word for word in jieba.cut(line, cut_all=False))]
            words =list(jieba.cut(line, cut_all=False))
            clean_pair.append(' '.join(word for word in words))
            #clean_pair.append(' '.join(HanziConv.toTraditional(word) for word in words))
        cleaned.append(clean_pair)
    return np.array(cleaned)

def save_clean_data(sentences, filename):
    with open(filename, 'wb') as f:
        dump(sentences, f)
        print('Saved: %s' % filename)

def save_clean_data2(sentences, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write("\"%s\",\"%s\",\"%s\"\n" % (sentence[0], sentence[1], sentence[2]))
        print('Saved: %s' % filename)

def main(argv):
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    dict_path = FLAGS.dict_path
    if dict_path == "" or dict_path == None:
        print('Use Default')
    else:
        print('Use Dict: ', dict_path)
        jieba.load_userdict(dict_path)

    datas = load_data(data_path)
    pairs = to_pairs(datas)
    clean_pairs = clean_pair(pairs)
    #save_clean_data(clean_pairs, output_path)
    print("Sentence Counts:", len(clean_pairs))
    for i in range(10):
        print('[%s]: [%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1], clean_pairs[i,2]))

if __name__ == '__main__':
    tf.app.run()
