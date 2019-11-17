from __future__ import print_function

from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import classification_report,accuracy_score
from attention import Position_Embedding, Attention
from keras.utils import to_categorical
import numpy as np
def get_search_data():
    # x_data = np.load("../../data/results.npy")
    # x_data = np.concatenate([x_data,np.load("../../data/descriptions.npy")],axis=-1)
    # y_data = np.load("../../data/labels_3_cat.npy")+1
    #
    # x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, train_size=0.8)



    data_1 = np.load('../../../data/results_big.npy')
    data_2 = np.load('../../../data/descriptions_big.npy')
    labels_all = to_categorical(np.load('../../../data/labels_3_cat_big.npy'))


    con_data=np.concatenate([data_1,data_2],axis=-1)

    split = int(len(data_1) * 9 / 10)

    data_1_train = data_1[0:split]
    data_2_train = data_2[0:split]
    con_data_train = con_data[0:split]


    data_1_test = data_1[split:]
    data_2_test = data_2[split:]
    con_data_test = con_data[split:]
    # con_data_test=np.concatenate([data_1_test,np.zeros(data_2_test.shape)],axis=-1)

    y_train = labels_all[:split]
    y_test = labels_all[split:]
    print(y_test.shape)



    return [con_data_train, con_data_test],[y_train, y_test]

[x_train, x_test],[y_train, y_test] = get_search_data()

# max_features = 20000
# maxlen = 80
# batch_size = 32
#
# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)

from keras.models import Model
from keras.layers import *

S_inputs = Input(shape=(x_train.shape[1],x_train.shape[2]))
# embeddings = Embedding(max_features, 128)(S_inputs)
# embeddings = Position_Embedding()(S_inputs)  # 增加Position_Embedding能轻微提高准确率
O_seq = Attention(16, 16)([S_inputs, S_inputs, S_inputs])
# O_seq=Attention(16, 16)([O_seq, O_seq, O_seq])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(3, activation='softmax')(O_seq)


model = Model(inputs=S_inputs, outputs=outputs)
print(model.summary())

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

for i in range(15):
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=1,
              validation_data=(x_test, y_test))

    y_pred = model.predict(x_test)
    y_pred_cat = np.round(y_pred)

    print(classification_report(y_test, y_pred_cat))

    # score, acc = model.evaluate(x_test, y_test, batch_size=256)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
