#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split


def BiLSTM(x_train, y_train):
#     max_features = 20000
#     # cut texts after this number of words
#     # (among top max_features most common words)
    maxlen = x_train.shape[1]
#     batch_size = 32

    model = Sequential()
#     model.add(Embedding(x_train.shape[-1], 100, input_length=maxlen))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
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
    y_data = np.load("../../data/labels.npy")
    
    x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, train_size=0.8)

    return [x_train, x_test],[y_train, y_test]
    


[x_train, x_test],[y_train, y_test] = get_search_data()
model = BiLSTM(x_train, y_train)
# model.summary()
print('Train...')
model.fit(x_train, y_train,
          epochs=15,
          validation_data=[x_test, y_test])

model.save('BiLSTM_2_cat.h5')







