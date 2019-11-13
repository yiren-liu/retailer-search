#!/usr/bin/env python
# coding: utf-8


import pickle
import numpy as np
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
#from keras.utils import to_categorical

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self): 
        self.terminal.flush() 
        self.log.flush()

sys.stdout = Logger()


def SVM():

    model = svm.SVC()

    return model



def gen_dummy_data(x=100,y=128,z=768):
    x_train = np.random.rand(x,y,z)
    y_train = np.array([np.random.randint(2) for i in range(x)])
    #y_train = to_categorical(y_train+1)
    x_test = np.random.rand(x,y,z)
    y_test = np.array([np.random.randint(2) for i in range(x)])
    #y_test = to_categorical(y_test+1)
    
    return [x_train, x_test],[y_train, y_test]
    
def get_search_data():
    # x_data = np.load("../../data/results.npy")
    # x_data = np.concatenate([x_data,np.load("../../data/descriptions.npy")],axis=-1)
    # y_data = np.load("../../data/labels_3_cat.npy")+1
    #
    # x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, train_size=0.8)

    data_1 = np.load('../../data/results_big_BoW.npy')
    data_2 = np.load('../../data/descriptions_big_BoW.npy')
    labels_all = np.load('../../data/labels_3_cat_big.npy')
    con_data=np.concatenate([data_1,data_2],axis=-1)

    split = int(len(data_1) * 4 / 5)
    # facet_train = facet_all[0:3000]
    data_1_train = data_1[0:split]
    data_2_train = data_2[0:split]
    con_data_train = con_data[0:split]

    # facet_test = facet_all[3000:]
    data_1_test = data_1[split:]
    data_2_test = data_2[split:]
    con_data_test = con_data[split:]

    y_train = labels_all[:split]
    y_test = labels_all[split:]
    print(y_test.shape)

    #only use results for testing
    data_1_test = np.concatenate([data_1_test,np.zeros(data_2_test.shape)],axis=-1)

    return [con_data_train, data_1_test],[y_train, y_test]


[x_train, x_test],[y_train, y_test] = get_search_data()
#[x_train, x_test],[y_train, y_test] = gen_dummy_data()


#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
model = SVM()
# model.summary()
print('Train...')
#print(x_train, y_train)

model.fit(x_train, y_train)


y_pred_cat = model.predict(x_test)

print(classification_report(y_test, y_pred_cat))
print("accuracy {:.2f}".format(accuracy_score(y_test, y_pred_cat)))


