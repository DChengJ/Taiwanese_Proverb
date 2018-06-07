import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.callbacks import TensorBoard
import numpy as np
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './data/nonbalance/data', 'path to directory')
tf.app.flags.DEFINE_string('Train_set', '0m_TR.txt', '')
tf.app.flags.DEFINE_string('Test_set', '0m_TE.txt', '')
tf.app.flags.DEFINE_integer('epochs_step', 20, 'epochs_step')
tf.app.flags.DEFINE_integer('_batch_size', 128, 'epochs_step')
tf.app.flags.DEFINE_integer('limit_v', 1, 'Word Vector Frequence > 1')
tf.app.flags.DEFINE_integer('num_classes', 3, 'Number of Labels')
tf.app.flags.DEFINE_boolean('remove_mid', False, 'Remove Label Mid')
tf.app.flags.DEFINE_integer('num_CV', 0, 'N-cross-validation')

# Setup parameters
# Model Saver Parameters
log_dir = "model/"

# Training Parameters
learning_rate = 0.1
result = {}

'''
# parameters for LSTM
nb_lstm_outputs = 1000   # Number of neurons
nb_time_steps = 1 #LSTM time step
activation = 'relu'
'''
dropout_rate = 0.7
#nb_input_vector = 0 # Input vector defualt 0
num_classes = 3 # Total classes (0-2)

# Self function
def LoadData(path, remove_mid):
    datas = []
    lables = []
    with open(path, "r") as f:
        for line in f:
            tmpint = int(re.split('[,]+', line.strip(), maxsplit=1)[0])
            #print(re.split(r"\s+|\t+", line.strip())[0])
            tmplist = []
            for i in range(num_classes):
                if remove_mid and i == 1:
                    continue
                if i == (tmpint-1):
                    tmplist.append(1)
                else:
                    tmplist.append(0)
            datas.append(re.split('[,]+', line.strip(), maxsplit=1)[1].split(","))
            lables.append(tmplist)
    return np.array(datas, np.int32), np.array(lables, np.int32)

# Find Limit Frequence
def find_limit(train_data, limit_v):
    index_list = []
    res_list = {}
    for times in train_data:
        for index, v in enumerate(times.tolist()):
            if not index in res_list:
                res_list[index] = v
            else:
                res = res_list[index]
                res_list[index] = res + v
    for i, v in res_list.items():
        if v >= limit_v:
            index_list.append(i)
    return index_list

def ref_data(data, index_list):
    datas = data.tolist()
    new_data = []
    for times in datas:
        res = []
        for index in index_list:
            res.append(times[index])
        new_data.append(res)
    return np.array(new_data, np.int32)

# Confusion Matrix
def self_confusionM(cor_list, pred_list):
    class_list = []
    cor_class = {}
    pred_class = {}
    True_class = {}
    totals = 0
    all_precision = all_recall = all_f1 = 0
    
    for index, cor_v in enumerate(cor_list):
        pred_v = pred_list[index]
        if cor_v == pred_v:
            if not cor_v in True_class:
                True_class[cor_v] = 1
            else:
                num = True_class[cor_v]
                True_class[cor_v] = (num + 1)
        # Calculation cor_list
        if not cor_v in cor_class:
            cor_class[cor_v] = 1
        else:
            num = cor_class[cor_v]
            cor_class[cor_v] = (num + 1)
        # Calculation pred_list
        if not pred_v in pred_class:
            pred_class[pred_v] = 1
        else:
            num = pred_class[pred_v]
            pred_class[pred_v] = (num + 1)
        # Calculation class_list 
        if not cor_v  in class_list:
            class_list.append(cor_v)
    # Check pred_class contain class_list all value
    for class_i in class_list:
        totals += cor_class[class_i]
        if not class_i in True_class:
            True_class[class_i] = 0
        if not class_i in pred_class:
            pred_class[class_i] = 0
    print('Multi Class Confusion Matrix')
    print('Class\tPrecision\tRecall\t\tF-Measure\tNumbers')
    for class_i in class_list:
        pc = float(pred_class[class_i])
        cc = float(cor_class[class_i])
        tc = float(True_class[class_i])
        if pc == 0:
            precision = 0
        else:
            precision = tc/pc
        if cc ==0:
            recall = 0
        else:
            recall = tc / cc
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        all_precision +=precision * cor_class[class_i] / totals
        all_recall += recall * cor_class[class_i] / totals
        all_f1 += f1 * cor_class[class_i] / totals
        print('%d\t%.3f\t\t%.3f\t\t%.3f\t\t%d' % (class_i, precision, recall, f1, cor_class[class_i]))
    print('---------------------------------------------------')
    print('%s\t%.3f\t\t%.3f\t\t%.3f\t\t%d' % ('all', all_precision, all_recall, all_f1, totals))
    return all_precision, all_recall, all_f1

def printLabelNum(labels_list):
    class_list = []
    labels_dict = {}
    for label in labels_list:
        if not label  in class_list:
            class_list.append(label)
        if not label in labels_dict:
            labels_dict[label] = 1
        else:
            num = labels_dict[label]
            labels_dict[label] = num + 1
    print("Number of Labels")
    class_list.sort()
    for class_i in class_list:
        print("%d\t%d" % (class_i, labels_dict[class_i]))
    print()

# build model
def build_model(nb_input_vector, num_classes):
    model = Sequential()
    model.add(Dense(1024, activation='tanh', input_dim=nb_input_vector))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(dropout_rate))
    #model.add(LSTM(units=nb_lstm_outputs, input_shape= (nb_time_steps, nb_input_vector), return_sequences=True))
    #model.add(Activation('tanh'))
    #model.add(Dropout(dropout_rate))
    #model.add(LSTM(nb_lstm_outputs, return_sequences=True))
    #model.add(Dropout(dropout_rate))
    #model.add(LSTM(nb_lstm_outputs))
    #model.add(Activation('tanh'))
    #model.add(Dropout(dropout_rate))    
    #model.add(Activation(activation))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def self_metric(truth_y, pred_y):
    # Metrics
    print("Metrics")
    '''
    Pred_Y = tf.placeholder(tf.int32, [None])
    GT_Y = tf.placeholder(tf.int32, [None])
    tfacc, acc_op = tf.metrics.accuracy(labels=GT_Y, predictions=Pred_Y)
    precision, pre_op = tf.metrics.precision(labels=GT_Y, predictions=Pred_Y)
    recall, recall_op = tf.metrics.recall(labels=GT_Y, predictions=Pred_Y)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    Acc = sess.run(acc_op, feed_dict={GT_Y: truth_y, Pred_Y: pred_y})
    Precision = sess.run(pre_op, feed_dict={GT_Y: truth_y, Pred_Y: pred_y})
    Recall = sess.run(recall_op, feed_dict={GT_Y: truth_y, Pred_Y: pred_y})
    print(Precision)
    '''
    sess = tf.Session()
    print(sess.run(tf.contrib.metrics.confusion_matrix(labels=truth_y, predictions=pred_y)))
    precision, recall, f1 = self_confusionM(truth_y, pred_y)
    return precision, recall, f1 

def run(num_CV, data_path, Train_name, Test_name, num_classes, epochs_step, _batch_size, remove_mid, limit_v):
    for n in range(num_CV):
        print('-------------------------------------------------------------------------------------------------')
        print(n+1)
        # data preprocessing: tofloat32, normalization, one_hot encoding
        Train_set = "%s/%s%s" % (data_path, n, Train_name)
        Test_set = "%s/%s%s" % (data_path, n, Test_name)

        # Train and Test Data Load
        x_train, y_train = LoadData(Train_set, remove_mid)
        x_test, y_test = LoadData(Test_set, remove_mid)
        print("Before Word Vector Len: " + str(len(x_train[0])))
        index_lists = find_limit(x_train, limit_v)
        x_train = ref_data(x_train, index_lists)
        x_test = ref_data(x_test, index_lists)
        nb_input_vector = len(x_train[0])
        #x_train = x_train.reshape([-1,nb_time_steps,nb_input_vector])
        #x_test = x_test.reshape([-1,nb_time_steps,nb_input_vector])
        num_examples = len(x_train)
        print("After Word Vector Len: " + str(nb_input_vector))
        if remove_mid: num_classes = 2

        save_path = "%s/%s%s%s" % (data_path, n, limit_v, "m_MODEL.h5")
        print(save_path)
        if not os.path.isfile(save_path):
            print('N')
            callback = TensorBoard(log_dir='./model', histogram_freq=0,  write_graph=True, write_images=True)
            model = build_model(nb_input_vector, num_classes)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # train: epcoch, batch_size
            print("Start Train")
            #model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, callbacks=[callback], verbose=2)
            model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, verbose=2)
            model.save(save_path)
        else:
            print('Y')
            model = tf.contrib.keras.models.load_model(save_path)
        print("End Train")
        #print("Start Predict")
        pred = model.predict(x_test, batch_size=_batch_size, verbose=0)
        _y = np.argmax(y_test, 1).tolist()
        pred = np.argmax(pred, 1).tolist()
        #printLabelNum(np.argmax(y_train, 1))
        #printLabelNum(np.argmax(y_test, 1))
        precision, recall, f1 = self_metric(_y, pred)

        #model.summary()
        print("Start Evaluate")
        score = model.evaluate(x_test, y_test,batch_size=_batch_size, verbose=1)
        print ("test accuracy: %.4f" % (score[1]))
        print("test F1: %.4f" % (f1))
        print("%.1f%%(%.2f)" % (round(score[1]*10000)/100, f1))

        del model

def main(argv):
    assert FLAGS.data_path and FLAGS.num_classes
    assert FLAGS.Train_set and FLAGS.Test_set
    assert FLAGS.epochs_step and FLAGS._batch_size
    num_classes = FLAGS.num_classes
    remove_mid = FLAGS.remove_mid
    data_path = FLAGS.data_path
    Train_set = FLAGS.Train_set
    Test_set = FLAGS.Test_set
    limit_v = FLAGS.limit_v
    epochs_step = FLAGS.epochs_step
    _batch_size = FLAGS._batch_size
    num_CV = FLAGS.num_CV

    if num_CV > 0:
        run(num_CV, data_path, Train_set, Test_set, num_classes, epochs_step, _batch_size, remove_mid, limit_v)
    else:
        # data preprocessing: tofloat32, normalization, one_hot encoding
        Train_set = "%s/%s" % (data_path, Train_set)
        Test_set = "%s/%s" % (data_path, Test_set)

        # Train and Test Data Load
        x_train, y_train = LoadData(Train_set, remove_mid)
        x_test, y_test = LoadData(Test_set, remove_mid)
        print("Before Word Vector Len: " + str(len(x_train[0])))
        index_lists = find_limit(x_train, limit_v)
        x_train = ref_data(x_train, index_lists)
        x_test = ref_data(x_test, index_lists)
        nb_input_vector = len(x_train[0])
        #x_train = x_train.reshape([-1,nb_time_steps,nb_input_vector])
        #x_test = x_test.reshape([-1,nb_time_steps,nb_input_vector])
        num_examples = len(x_train)
        print("After Word Vector Len: " + str(nb_input_vector))
        if remove_mid: num_classes = 2

        callback = TensorBoard(log_dir='./model', histogram_freq=0,  write_graph=True, write_images=True)
        model = build_model(nb_input_vector, num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train: epcoch, batch_size
        print("Start Train")
        #model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, callbacks=[callback], verbose=2)
        model.fit(x_train, y_train, epochs=epochs_step, batch_size=_batch_size, verbose=2)
        print("End Train")
        #print("Start Predict")
        pred = model.predict(x_test, batch_size=_batch_size, verbose=0)
        _y = np.argmax(y_test, 1).tolist()
        pred = np.argmax(pred, 1).tolist()

        printLabelNum(np.argmax(y_train, 1))
        printLabelNum(np.argmax(y_test, 1))
        precision, recall, f1 = self_metric(_y, pred)

        model.summary()
        print("Start Evaluate")
        score = model.evaluate(x_test, y_test,batch_size=_batch_size, verbose=1)
        print ("test accuracy: %.4f" % (score[1]))
        print("test F1: %.4f" % (f1))
        print("%.1f%%(%.2f)" % (round(score[1]*10000)/100, f1))

if __name__ == '__main__':
    tf.app.run()
        
        
