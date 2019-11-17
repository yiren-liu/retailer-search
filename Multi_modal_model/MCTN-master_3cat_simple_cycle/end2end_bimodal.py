import os
import sys
sys.path.extend(['models'])
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# from keras import backend as K

from models.bimodals import E2E_MCTN_Model
from utils.args import parse_args
from utils.utils import get_preds_statistics
from utils.data_loader import load_and_preprocess_data,load_search_data
import pickle
from keras.models import load_model
from sklearn.metrics import classification_report,accuracy_score



def draw_loss_pic(history):
    import matplotlib.pyplot as plt
    with open('history_params.sav', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    history_dict = history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')  # ←------'bo'
    #表示蓝色圆点
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # ←------'b'
    #表示蓝色实线
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()






np.random.seed(123)
tf.set_random_seed(456)

# arguments
args, configs = parse_args()

# data load
is_cycled = configs['translation']['is_cycled']
feats_dict = load_search_data()

print("FORMING SEQ2SEQ MODEL...")
features = args.feature  # e.g. ['a', 't']
assert len(features) == 2, 'Wrong number of features'
end2end_model = E2E_MCTN_Model(configs, features, feats_dict)

print("PREP FOR TRAINING...")
filename = '_'.join(args.feature) + "_attention_seq2seq_" + \
           str("bi_directional" if configs['translation']['is_bidirectional']
               else '') + \
           "_bimodal.h5"

output_dir = configs['general']['output_dir']
weights_path = os.path.join(output_dir, filename)
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

callbacks = [
  # EarlyStopping(monitor='val_loss', patience=args.train_patience, verbose=0),
  ModelCheckpoint('model.h5', monitor='val_categorical_accuracy', save_best_only=True,
                  verbose=1),
]

# try:
#     end2end_model.model.load_weights(weights_path)
#     print("\nWeights loaded from {}\n".format(weights_path))
# except:
#     print("\nCannot load weight. Training from scratch\n")


#
# print("TRAINING NOW...")
# train=0
#
#
# if train==1:
#     history=end2end_model.train(weights_path=weights_path,
#                         n_epochs=args.train_epoch,
#                         val_split=args.val_split,
#                         batch_size=args.batch_size,
#                         callbacks=callbacks
#                                 )
#
#     with open('history_params.sav', 'wb') as f:
#         pickle.dump(history.history, f, -1)
# else:
#     end2end_model.model.load_weights('model.h5')
#
# print("PREDICTING...")
# predictions = end2end_model.predict()
# # predictions = predictions.reshape(-1, )
# get_preds_statistics(predictions, feats_dict['test_labels'])


for i in range(args.train_epoch):
    history = end2end_model.train(weights_path=weights_path,
                                  n_epochs=1,
                                  val_split=args.val_split,
                                  batch_size=args.batch_size,
                                  callbacks=callbacks
                                  )
    predictions = end2end_model.predict()
    # predictions = predictions.reshape(-1, )
    get_preds_statistics(predictions, feats_dict['test_labels'])


def get_10_fold_data(X, Y):
    seed = 7

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    return kfold.split(X, Y)

data_1,data_2,labels_all,num_labels=load_data_for_cross_test()
flag = 0
for train, test in get_10_fold_data(data_1, num_labels):
    flag+=1
    data_1_train=data_1[train]
    data_2_train = data_2[train]
    y_train = labels_all[train]

    data_1_test = data_1[test]
    data_2_test = data_2[test]
    y_test = labels_all[test]
    end2end_model = E2E_MCTN_Model(configs, features, feats_dict)

    for i in range(15):
        print('now it is epoch %d'%i)
        end2end_model.model.fit(x=[data_1_train],
                  y=[data_2_train.mean(1),y_train],
                  epochs=1,
                  validation_data=[data_1_test, [data_2_test.mean(1), y_test]],
                  # self.input_test,
                  batch_size=256,
                  verbose=2,)
        predictions = end2end_model.model.predict(data_1_test)[-1]
        get_preds_statistics(predictions, feats_dict['test_labels'])

    print('--------------------------finish %d cross validation------------------------------------------------'%flag)
