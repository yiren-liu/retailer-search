#!/usr/bin/env python
# coding: utf-8

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from thundersvm import SVC
from sklearn.model_selection import StratifiedKFold
#from keras.utils import to_categorical

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self): 
        self.terminal.flush() 
        self.log.flush()

sys.stdout = Logger()


def SVM():

    model = SVC()

    return model

def get_10_fold_data(X, Y):
    seed = 7
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    return kfold.split(X, Y)

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
    n=1
    data_1 = np.load('../../data/results_big_BoW_%d_gram.npy'%n)
    data_2 = np.load('../../data/descriptions_big_BoW_%d_gram.npy'%n)
    labels_all = np.load('../../data/labels_3_cat_big.npy')#to_categorical
    con_data=np.concatenate([data_1,data_2],axis=-1)

    split = int(len(data_1) *9 / 10)
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
    # data_1_test = np.concatenate([data_1_test,np.zeros(data_2_test.shape)],axis=-1)
    data_1_train = data_1_train + data_2_train
    return [data_1_train, data_1_test],[y_train, y_test]


# [x_train, x_test],[y_train, y_test] = get_search_data()
# #[x_train, x_test],[y_train, y_test] = gen_dummy_data()


# #y_train = to_categorical(y_train)
# #y_test = to_categorical(y_test)
# model = SVM()


# # model.summary()
# print('Train...')
# #print(x_train, y_train)

# model.fit(x_train, y_train)


# y_pred_cat = model.predict(x_test)

# print(classification_report(y_test, y_pred_cat))
# print("accuracy {:.2f}".format(accuracy_score(y_test, y_pred_cat)))


n=1
data_1 = np.load('../../data/results_big_BoW_%d_gram.npy'%n)
data_2 = np.load('../../data/descriptions_big_BoW_%d_gram.npy'%n)
labels_all = np.load('../../data/labels_3_cat_big.npy')#to_categorical
con_data=np.concatenate([data_1,data_2],axis=-1)


flag = 0
avg_dict = {'micro avg': {'precision': 0,'recall': 0,'f1-score':0}, 'micro avg': {'precision': 0,'recall': 0,'f1-score':0}}
for train, test in get_10_fold_data(con_data, labels_all):
    flag+=1

    x_train = con_data[train]
    y_train = labels_all[train]
    x_test = con_data[test]
    y_test = labels_all[test]


    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    model = SVM()

    print('fold: '+ str(flag))
    print('results + descriptions ---> results + descriptions')
    # model.summary()
    print('Train...')
    #print(x_train, y_train)

    model.fit(x_train, y_train)


    y_pred_cat = model.predict(x_test)

    print(classification_report(y_test, y_pred_cat))
    print("accuracy {:.2f}".format(accuracy_score(y_test, y_pred_cat)))

    temp_dict = classification_report(y_test, y_pred_cat, output_dict = True) 
    for key1 in avg_dict:
        for key2 in avg_dict[key1]:
            avg_dict[key1][key2] += temp_dict[key1][key2]
            
for key1 in avg_dict:
    for key2 in avg_dict[key1]:
        avg_dict[key1][key2] = avg_dict[key1][key2]/10

print("average score for results + descriptions ---> results + descriptions:")
print(avg_dict)



flag = 0
avg_dict = {'micro avg': {'precision': 0,'recall': 0,'f1-score':0}, 'micro avg': {'precision': 0,'recall': 0,'f1-score':0}}
for train, test in get_10_fold_data(con_data, labels_all):
    flag+=1

    x_train = con_data[train]
    y_train = labels_all[train]


    x_test = np.concatenate([data_1[test],np.zeros(data_2[test].shape)],axis=-1)
    y_test = labels_all[test]

    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    model = SVM()

    print('fold: '+ str(flag))
    print('results + descriptions ---> results')
    # model.summary()
    print('Train...')
    #print(x_train, y_train)

    model.fit(x_train, y_train)


    y_pred_cat = model.predict(x_test)

    print(classification_report(y_test, y_pred_cat))
    print("accuracy {:.2f}".format(accuracy_score(y_test, y_pred_cat)))

    temp_dict = classification_report(y_test, y_pred_cat, output_dict = True) 
    for key1 in avg_dict:
        for key2 in avg_dict[key1]:
            avg_dict[key1][key2] += temp_dict[key1][key2]

for key1 in avg_dict:
    for key2 in avg_dict[key1]:
        avg_dict[key1][key2] = avg_dict[key1][key2]/10

print("average score for results + descriptions ---> results:")
print(avg_dict)