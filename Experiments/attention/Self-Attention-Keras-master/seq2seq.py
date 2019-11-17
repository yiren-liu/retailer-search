from __future__ import print_function

from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import classification_report,accuracy_score
from attention import Position_Embedding, Attention
from keras.utils import to_categorical
import numpy as np
from cells import AttentionDecoderCell
from cells import LSTMDecoderCell
from keras.models import Model
from keras.layers import *
from recurrentshop import LSTMCell
from recurrentshop import RecurrentSequential

#params
dropout=0.5
hidden_dim=256
depth=[1,1]

def seq2seq_model(x_train_1,x_train_2):
    #encoder
    S_inputs = Input(shape=(x_train_1.shape[1], x_train_1.shape[2]))
    # embeddings = Embedding(max_features, 128)(S_inputs)
    # embeddings = Position_Embedding()(S_inputs)  # 增加Position_Embedding能轻微提高准确率
    encoded = Attention(32, 32)([S_inputs, S_inputs, S_inputs])
    # O_seq=Attention(16, 16)([O_seq, O_seq, O_seq])
    # O_seq = GlobalAveragePooling1D()(O_seq)
    # O_seq = Dropout(dropout)(O_seq)
    # outputs = Dense(3, activation='softmax')(O_seq)


    #decoder
    decoder = RecurrentSequential(decode=True, output_length=1,  # x_train_2.shape[1]
                                  unroll=False, stateful=False)
    decoder.add(
        Dropout(dropout, batch_input_shape=(None, x_train_1.shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(
            AttentionDecoderCell(output_dim=x_train_2.shape[2], hidden_dim=hidden_dim))
    else:
        decoder.add(
            AttentionDecoderCell(output_dim=x_train_2.shape[2], hidden_dim=hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=x_train_2.shape[2], hidden_dim=hidden_dim))


    #regression model
    x= Attention(8, 16)([encoded, encoded, encoded])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    regr_outputs = Dense(3, activation='softmax')(x)


    decoded = decoder(encoded)
    decoded = Reshape((x_train_2.shape[2],))(decoded)
    model = Model(inputs=S_inputs, outputs=[decoded,regr_outputs])
    print(model.summary())

    # try using different optimizers and different optimizer configs
    model.compile(loss=['mse','categorical_crossentropy'],loss_weights=[1,10], optimizer='adam', metrics=['categorical_accuracy'])

    return model

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



    return [data_1_train, data_1_test],[data_2_train,data_2_test],[y_train, y_test]

if __name__=='__main__':
    [data_1_train, data_1_test],[data_2_train,data_2_test],[y_train, y_test] = get_search_data()
    model=seq2seq_model(data_1_train,data_2_train)

    for i in range(15):
        print('Train...')
        model.fit(data_1_train, [data_2_train.mean(1),y_train],
                  batch_size=256,
                  epochs=1,
                  validation_data=(data_1_test, [data_1_test.mean(1),y_test]))

        y_pred = model.predict(data_1_test)[-1]
        y_pred_cat = np.round(y_pred)

        print(classification_report(y_test, y_pred_cat))

        # score, acc = model.evaluate(x_test, y_test, batch_size=256)
        # print('Test score:', score)
        # print('Test accuracy:', acc)