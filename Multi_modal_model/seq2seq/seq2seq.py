from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.utils import to_categorical


class Seq2seq_Model(object):

    def __init__(self, dataset_dict):
        self._create_model()
        self.input_train = dataset_dict['res'][0]
        self.input_test = dataset_dict['res'][1]
        self.output_train = dataset_dict['des'][0]
        self.output_test = dataset_dict['des'][1]

    def _create_model(self):
        input_dims = self.input_train.shape
        output_dims = self.output_train.shape

        # --------------------Seq2Seq network params--------------------------------
        n_samples = input_dims[0]
        input_length = input_dims[1]
        input_dim = input_dims[2]
        output_length = output_dims[1]
        output_dim = output_dims[2]
        encoder_inputs,decoder_inputs,decoder_outputs=self.seq2seq_model()

    def seq2seq_model(self,input_length,input_dim,output_length,output_dim,latent_dim = 256 ):

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, input_dim))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, output_dim))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(output_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return encoder_inputs,decoder_inputs,decoder_outputs

    def classification_model(self):
        pass


def load_data():
    data_1 = np.load('../../data/results_big.npy')
    data_2 = np.load('../../data/descriptions_big.npy')
    labels_all = to_categorical(np.load('../../data/labels_3_cat_big.npy'))
    split = int(len(data_1) * 4 / 5)
    # facet_train = facet_all[0:3000]
    data_1_train = data_1[0:split]
    data_2_train = data_2[0:split]

    # facet_test = facet_all[3000:]
    data_1_test = data_1[split:]
    data_2_test = data_2[split:]

    y_train = labels_all[:split]
    y_test = labels_all[split:]

    dataset_dict = {'res': [data_1_train, data_1_test],
                  # 'f': [facet_train, facet_test, 'video'],
                  'des': [data_2_train, data_2_test],
                  'train_labels': y_train,
                  'test_labels': y_test
                  }
    return dataset_dict

if __name__=='__main__':
    dataset_dict=load_data()

    model=Seq2seq_Model(dataset_dict)