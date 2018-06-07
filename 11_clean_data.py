import tensorflow as tf
import numpy as np
import collections
import string
import jieba
import re
from pickle import dump
from hanziconv import HanziConv

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './data/newProverb2_v4_0.csv', 'path to directory')
tf.app.flags.DEFINE_string('dict_path', './tools/extra_dict/dict.txt', 'dict path to directory')
tf.app.flags.DEFINE_string('output_path', './pkl_data/newProverb2_jieba_v4_0.pkl', 'output path to directory')


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
            line = HanziConv.toSimplified(line)
            # line = [' '.join(word for word in jieba.cut(line, cut_all=False))]
            words =list(jieba.cut(line, cut_all=False))
            clean_pair.append(' '.join(HanziConv.toTraditional(word) for word in words))            
        cleaned.append(clean_pair)
    return np.array(cleaned)

def save_clean_data(sentences, filename):
    with open(filename, 'wb') as f:
        dump(sentences, f)
        print('Saved: %s' % filename)

def main(argv):
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    dict_path = FLAGS.dict_path
    if dict_path != "":
        print('Use Dict: ', dict_path)
        jieba.load_userdict(dict_path)
    else:
        print('Use Default')

    datas = load_data(data_path)
    pairs = to_pairs(datas)
    clean_pairs = clean_pair(pairs)
    save_clean_data(clean_pairs, output_path)
    for i in range(10):
        print('[%s]: [%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1], clean_pairs[i,2]))

if __name__ == '__main__':
    tf.app.run()
