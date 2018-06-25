import jieba
import logging
import re
from hanziconv import HanziConv

class Segmentation(object):
    def __init__(self):
        logging.basicConfig(format = "%(asctime)s : %(levelname)s : %(message)s", level = logging.INFO)
    
    def process_text(self):
        logging.info("等待中..(簡 to 繁)")
        with open('./word2vec_data/traditional.txt', 'w', encoding='utf-8') as fw :
            with open('./word2vec_data/wiki_text.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = HanziConv.toTraditional(line)
                    fw.write(line)
                    
    def segmentation(self):
        rule = re.compile(r'[^\u4e00-\u9fa5]')
        jieba.set_dictionary('./tools/extra_dict/dict.txt')
        with open('./word2vec_data/segmentation.txt', 'w', encoding='utf-8') as segmentation :
            with open('./word2vec_data/traditional.txt', 'r', encoding='utf-8') as Corpus:
                for sentence in Corpus:
                    sentence = sentence.strip("\n")
                    sentence = rule.sub("", sentence)
                    words =list(jieba.cut(sentence, cut_all=False))
                    segmentation.write(' '.join(words))
        
if __name__ == "__main__":
    segmentation = Segmentation()
    segmentation.process_text()
    segmentation.segmentation()
