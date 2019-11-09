#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import classification_report

def BiLSTM(x_train, y_train):
#     max_features = 20000
#     # cut texts after this number of words
#     # (among top max_features most common words)
    maxlen = x_train.shape[1]
#     batch_size = 32

    model = Sequential()
#     model.add(Embedding(x_train.shape[-1], 100, input_length=maxlen))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['binary_accuracy'])
    return model


def gen_imdb_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return [x_train, x_test],[y_train, y_test]

def gen_dummy_data(x=761,y=128,z=768):
    x_train = np.random.rand(x,y,z)
    y_train = np.array([np.random.randint(2)*2-1 for i in range(x)])
    x_test = np.random.rand(x,y,z)
    y_test = np.array([np.random.randint(2)*2-1 for i in range(x)])
    
    return [x_train, x_test],[y_train, y_test]
    
def get_search_data():
    x_data = np.load("../../data/results.npy")
    x_data = np.concatenate([x_data,np.load("../../data/descriptions.npy")],axis=-1)
    y_data = np.load("../../data/labels_3_cat.npy")+1
    
    x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, train_size=0.8)

    return [x_train, x_test],[y_train, y_test]
    
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
model = BiLSTM(x_train, y_train)
# model.summary()
print('Train...')
model.fit(x_train, y_train,
          epochs=15,
          validation_data=[x_test, y_test])

model.save('BiLSTM_3_cat.h5')


y_pred = model.predict(x_test)
y_pred_cat = np.round(y_pred)

print(classification_report(y_test, y_pred_cat))
print(y_pred_cat)



