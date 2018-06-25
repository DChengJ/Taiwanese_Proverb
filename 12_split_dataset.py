from pickle import load, dump
from numpy.random import rand
from numpy.random import shuffle
import numpy as np
import tensorflow as tf
import re
np.random.seed(2)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_CV', 5, 'N-cross-validation')
tf.app.flags.DEFINE_string('data_path', './pkl_data/newProverb2_jieba_v1_0.pkl', 'path to directory')
tf.app.flags.DEFINE_string('output_path', './crossvalidation', 'path to directory')

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

def save_clean_data2(sentences, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write("\"%s\",\"%s\",\"%s\"\n" % (sentence[0], sentence[1], sentence[2]))
        print('Saved: %s' % filename)

def calu_classes(lines):
    classes = dict()
    class_data = dict()
    for line in lines:
        label = int(line[0])
        if label not in classes:
            classes[label] = 1
            class_data[label] = list()
            class_data[label].append(list(line))
        else:
            classes[label] = classes[label] + 1
            class_data[label].append(list(line))
    '''
    for key in sorted(classes.keys()):
        add_num = classes[key] - int(classes[key]/5)*5
        print(key, classes[key], int(classes[key]/5), add_num)
    '''
    return classes, class_data

def split_data(num_CV, classes, dataset, data_path, output_path):
    CVdata= [{'train': list(), 'test': list()} for i in range(num_CV)]
    for label, counts in classes.items():
        epoch = int(counts / 5)
        stopadd = counts - epoch * 5
        print(counts)
        #print(epoch)
        s = 0
        e = epoch
        for i in range(num_CV):
            if i < stopadd: e = e + 1
            for j in range(num_CV):
                if i == j:
                    CVdata[j]["test"].extend(dataset[label][s:e])
                else:
                    CVdata[j]["train"].extend(dataset[label][s:e])
            print("%s, counts: %s, start: %s, end: %s" % (label, (e - s), s, e))
            s = e
            e = e + epoch
    return CVdata
    '''
    test_data = list()
    train_data = list()
    for num in range(num_CV):
        print()
        for label in sorted(classes):
            counts = classes[label]
            epoch = int(counts/5)
            test_s = num * epoch
            test_e = (num + 1) * epoch
            train_
            if num < (counts - epoch*5):
                print("Add")
            print(num, label, counts, (counts - epoch*5), s, e)
            print(len(dataset[label][test_s:test_e]))
            #print(classes[label])
    #save_clean_data(dataset, both_path)
    #print(dataset[1][0-2])
    '''
        
def outCV(CVdata, dataset, data_path, output_path):
        filename = data_path.rsplit('/', 1)[-1].split('.', 1)[0]
        both_path = "%s/%s-both.pkl" % (output_path, filename)
        save_clean_data(dataset, both_path)
        for i in range(len(CVdata)):
                train_filename = "%s/%s-%s-train" % (output_path, filename, (i + 1))
                test_filename = "%s/%s-%s-test" % (output_path, filename, (i + 1))
                print(train_filename)
                print(test_filename)
                for key, value in CVdata[i].items():
                        value = np.array(value)
                        print(key, ': ', len(value))
                        if key == 'train':
                                #save_clean_data(value, train_filename + '.pkl')
                                save_clean_data2(value, train_filename + '.csv')
                        else:
                                #save_clean_data(value, test_filename + '.pkl')
                                save_clean_data2(value, test_filename + '.csv')

def main(argv):
        data_path = FLAGS.data_path
        num_CV = FLAGS.num_CV
        output_path = FLAGS.output_path
        # load dataset
        raw_dataset = load_clean_sentences(data_path)
        dataset = raw_dataset
        shuffle(dataset)
        classes, class_data = calu_classes(dataset)
        CVdata = split_data(num_CV, classes, class_data, data_path, output_path)
        outCV(CVdata, dataset, data_path, output_path)

        '''
        n_sentences = len(raw_dataset)
        print(n_sentences)
        train_size = n_sentences * 0.8
        train_size = int(train_size)
        dataset = raw_dataset

        #train, test = dataset[:train_size], dataset[train_size:]
        #save_clean_data(dataset, './crossvalidation/newProverb2_jieba_v2_1-both.pkl')
        #save_clean_data(train, './crossvalidation/newProverb2_jieba_v2_1-train.pkl')
        #save_clean_data(test, './crossvalidation/newProverb2_jieba_v2_1-test.pkl')
        '''

if __name__ == '__main__':
        tf.app.run()
