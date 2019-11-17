from keras import Model
from seq2seq.mctn_models import mctn_model

from optimizers import get_optimizers
from regression_model import create_regression_model
from keras.layers import *

class BaseModel(object):
  def __init__(self, configs, features, feats_dict):
    self.configs = configs
    self.features = features
    self.feats_dict = feats_dict

    # to be updated by child classes
    self.model = None


class E2E_MCTN_Model(BaseModel):
  """End to end MCTN Bimodal, also a wrapper for data"""

  def __init__(self, configs, data_1_train,data_2_train):
    super(E2E_MCTN_Model, self).__init__(configs, data_1_train,data_2_train)



    # update self.model object
    self.is_cycled = configs['translation']['is_cycled']
    self._create_model()

  def _create_model(self):


    input_dim = 200
    output_dim = 200

    # --------------------Seq2Seq network params--------------------------------

    input_length = 100
    output_length = 100

    inputs=Input(shape=(input_length,))
    embeddings = Embedding(20000, input_dim)(inputs)
    # --------------- MODEL TRANSLATION DEFINITION -----------------------------
    print("Creating TRANSLATION SEQ2SEQ model ...")
    encoded_seq, decoded_seq, cycled_decoded_seq = \
      mctn_model(input=embeddings,output_dim=output_dim,
                 hidden_dim=self.configs['translation']['hidden_dim'],
                 output_length=output_length,
                 input_dim=input_dim,
                 input_length=input_length,
                 depth=self.configs['translation']['depth'],
                 bidirectional=self.configs['translation']['is_bidirectional'],
                 is_cycled=self.is_cycled,
                 dropout=0.5
                 )

    # ---------------- MODEL REGRESSION DEFINITION -----------------------------
    print("Creating REGRESSION model ...")
    regression_score = \
      create_regression_model(
        n_hidden=self.configs['regression']['reg_hidden_dim'],
        input=encoded_seq,
        l2_factor=self.configs['regression']['l2_factor'],
      input_hidden=self.configs['translation']['hidden_dim'],
      dropout=0.5)

    # ------------------ E2E REGRESSION DEFINITION -----------------------------
    print("BUILDING A JOINT END-TO-END MODEL")
    if self.is_cycled:
      outputs = [decoded_seq, cycled_decoded_seq, regression_score]
      losses = [self.configs['translation']['loss_type'],
                self.configs['translation']['cycle_loss_type'],
               self.configs['regression']['loss_type']
                ]
      losses_weights = [self.configs['translation']['loss_weight'],
                        self.configs['translation']['cycle_loss_weight'],
                        self.configs['regression']['loss_weight']
                        ]
    else:
      outputs = [decoded_seq, regression_score]
      losses = [self.configs['translation']['loss_type'],
                self.configs['regression']['loss_type']
                ]
      losses_weights = [self.configs['translation']['loss_weight'],
                        self.configs['regression']['loss_weight']
                        ]

    end2end_model = Model(inputs=inputs,
                          outputs=outputs)
    print("Compiling model")
    optimizer = get_optimizers(opt=self.configs['general']['optim'],
                               init_lr=self.configs['general']['init_lr'])
    end2end_model.compile(loss=losses,
                          loss_weights=losses_weights,
                          optimizer=optimizer,
                          metrics=[ 'categorical_accuracy'])
    # print("Model summary:")
    print(end2end_model.summary())
    # print("END2END MODEL CREATED!")

    self.model = end2end_model
    self.embeding_model=Model(inputs=inputs,
                          outputs=[embeddings])




