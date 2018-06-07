import tensorflow as tf
import numpy as np
import collections
import jieba
import os
import re

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './newProverb2_v1_0.csv', 'path to directory')
tf.app.flags.DEFINE_string('output_path', './newProverb2_jieba_v1_1.csv', 'output file path to directory')

def LoadData(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            #label, text1, text2 = re.split(',', line)
            label, text1, text2 = list(filter(None, re.split(r',|\"', line)))
            text1 = processText(text1)
            text2 = processText(text2)
            text = "%s,\"%s\",\"%s\"" % (label, text1, text2)
            datas.append(text)
    return datas
            
def processText(text):
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    text = rule.sub('', text)
    text = jieba_segment(text)
    return text

def jieba_segment(sentence):
    words = list(jieba.cut(sentence, cut_all=False))
    return ' '.join(words)

def outputFile(datas, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in datas:
            f.write(line + "\n")

def main(_):
    assert FLAGS.data_path and FLAGS.output_path
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    datas = LoadData(data_path)
    outputFile(datas, output_path)

if __name__ == '__main__':
    tf.app.run()
