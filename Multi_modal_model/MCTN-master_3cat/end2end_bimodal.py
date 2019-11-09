import os
import sys
sys.path.extend(['/home/zhengjie/Projects/Search/MCTN-master', '/home/zhengjie/Projects/Search/MCTN-master/models'])
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# from keras import backend as K

from models.bimodals import E2E_MCTN_Model
from utils.args import parse_args
from utils.utils import get_preds_statistics
from utils.data_loader import load_and_preprocess_data
import pickle
from sklearn.metrics import classification_report,accuracy_score



def draw_loss_pic(history):
    import matplotlib.pyplot as plt
    with open('trainHistoryDict', 'wb') as file_pi:
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
feats_dict = \
  load_and_preprocess_data(max_seq_len=configs['general']['max_seq_len'],
                           train_split=configs['general']['train_split'],
                           is_cycled=is_cycled)

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
  EarlyStopping(monitor='val_loss', patience=args.train_patience, verbose=0),
  ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True,
                  verbose=1),
]

print("TRAINING NOW...")
for i in range(50):
    end2end_model.train(weights_path=weights_path,
                        n_epochs=args.train_epoch,
                        val_split=args.val_split,
                        batch_size=args.batch_size,
                        callbacks=callbacks)
    # histroy=end2end_model.train(weights_path=weights_path,
    #                     n_epochs=args.train_epoch,
    #                     val_split=args.val_split,
    #                     batch_size=args.batch_size,
    #                     callbacks=callbacks)
    # draw_loss_pic(histroy.history)

    print("PREDICTING...")
    predictions = end2end_model.predict()
    # predictions = predictions.reshape(-1, )

    print(len(predictions))
    y_pred_cat = np.round(predictions)

    print(classification_report(feats_dict['test_labels'],y_pred_cat ))
    print(accuracy_score(feats_dict['test_labels'],y_pred_cat ))
    # print(y_pred_cat)
    # get_preds_statistics(y_pred_cat, feats_dict['test_labels'])
