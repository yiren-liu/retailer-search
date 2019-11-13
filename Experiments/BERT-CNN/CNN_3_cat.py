#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pickle
import numpy as np
import sys

from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

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



def CNN(x_train, y_train):
#     max_features = 20000
#     # cut texts after this number of words
#     # (among top max_features most common words)
    maxlen = x_train.shape[1]
    embedding_dim = x_train.shape[-1]
#     batch_size = 32

    model = Sequential()
#     model.add(Embedding(x_train.shape[-1], 100, input_length=maxlen))
    model.add(Conv1D(filters = int(np.round(maxlen/3)),kernel_size = 3, input_shape = (maxlen, embedding_dim)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['categorical_accuracy'])
    return model



def gen_dummy_data(x=761,y=128,z=768):
    x_train = np.random.rand(x,y,z)
    y_train = np.array([np.random.randint(2)*2-1 for i in range(x)])
    y_train = to_categorical(y_train+1)
    x_test = np.random.rand(x,y,z)
    y_test = np.array([np.random.randint(2)*2-1 for i in range(x)])
    y_test = to_categorical(y_test+1)
    
    return [x_train, x_test],[y_train, y_test]
    
def get_search_data():
    # x_data = np.load("../../data/results.npy")
    # x_data = np.concatenate([x_data,np.load("../../data/descriptions.npy")],axis=-1)
    # y_data = np.load("../../data/labels_3_cat.npy")+1
    #
    # x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, train_size=0.8)

    data_1 = np.load('../../data/results_big.npy')
    data_2 = np.load('../../data/descriptions_big.npy')
    labels_all = to_categorical(np.load('../../data/labels_3_cat_big.npy'))
    con_data=np.concatenate([data_1,data_2],axis=-1)

    split = int(len(data_1) * 9 / 10)
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
    
#---------------------------metrics---------------------------------------------#
def recall_m(y_true, y_pred):
        #true_positives = K.sum(K.cast(K.equal(K.round(K.clip(y_pred, 0, 1)),K.round(K.clip(y_true, 0, 1))),'float32'))
        true_positives = K.sum(K.round(K.clip(y_true, 0, 1) * K.clip(y_pred, 0, 1)))                
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        #true_positives = K.sum(K.cast(K.equal(K.round(K.clip(y_pred, 0, 1)),K.round(K.clip(y_true, 0, 1))),'float32'))
        true_positives = K.sum(K.round(K.clip(y_true, 0, 1) * K.clip(y_pred, 0, 1))) 
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




[x_train, x_test],[y_train, y_test] = get_search_data()
#[x_train, x_test],[y_train, y_test] = gen_dummy_data()

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
model = CNN(x_train, y_train)
# model.summary()
print('Train...')
callbacks = [
  # EarlyStopping(monitor='val_loss', patience=args.train_patience, verbose=0),
  ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True,
                  verbose=1),
]
history=model.fit(x_train, y_train,
          epochs=15,
          validation_data=[x_test, y_test], callbacks=callbacks)
with open('history_params.sav', 'wb') as f:
    pickle.dump(history.history, f, -1)

model = load_model('model.h5')

y_pred = model.predict(x_test)
y_pred_cat = np.round(y_pred)

print(classification_report(y_test, y_pred_cat))
print("accuracy {:.2f}".format(accuracy_score(y_test, y_pred_cat)))




