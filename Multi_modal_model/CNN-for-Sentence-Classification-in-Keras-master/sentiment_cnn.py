"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""
from keras.utils import to_categorical
import numpy as np
import data_helpers
# from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from keras.models import load_model
np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "keras_data_set"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3,4,5)
num_filters = 100
dropout_prob = (0.5, 0.8)
hidden_dims = 512

# Training parameters
batch_size = 256
num_epochs = 50

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------


def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.9)
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_test = x[train_len:]
        y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv
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


    return [data_1_train, data_1_test],[y_train, y_test]

# # Data Preparation
# print("Load data...")
# x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source)
#
# if sequence_length != x_test.shape[1]:
#     print("Adjusting sequence length for actual size")
#     sequence_length = x_test.shape[1]
#
# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)
# print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
#
# # Prepare embedding layer weights and convert inputs for static model
# print("Model type is", model_type)
# if model_type in ["CNN-non-static", "CNN-static"]:
#     embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
#                                        min_word_count=min_word_count, context=context)
#     if model_type == "CNN-static":
#         x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
#         x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
#         print("x_train static shape:", x_train.shape)
#         print("x_test static shape:", x_test.shape)
#
# elif model_type == "CNN-rand":
#     embedding_weights = None
# else:
#     raise ValueError("Unknown model type")
#
# # Build model
# if model_type == "CNN-static":
#     input_shape = (sequence_length, embedding_dim)
# else:
#     input_shape = (sequence_length,)
#
# model_input = Input(shape=input_shape)
#
# # Static model does not have embedding layer
# if model_type == "CNN-static":
#     z = model_input
# else:
#     z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)


[x_train, x_test],[y_train, y_test] = get_search_data()
model_input = Input(shape=(x_train.shape[1], x_train.shape[2]))
z=model_input
z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(3, activation='softmax')(z)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

# # Initialize weights with word2vec
# if model_type == "CNN-non-static":
#     weights = np.array([v for v in embedding_weights.values()])
#     print("Initializing embedding layer with word2vec weights, shape", weights.shape)
#     embedding_layer = model.get_layer("embedding")
#     embedding_layer.set_weights([weights])

# Train the model
model.summary()
callbacks = [
  # EarlyStopping(monitor='val_loss', patience=args.train_patience, verbose=0),
  ModelCheckpoint('CNN_3_cat.h5', monitor='val_loss', save_best_only=True,
                  verbose=1),
]
train=1
if train==1:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
              validation_data=(x_test, y_test), verbose=2,
                          callbacks=callbacks)
else:
    model = load_model('CNN_3_cat.h5')
y_pred = model.predict(x_test)
y_pred_cat = np.round(y_pred)

print(classification_report(y_test, y_pred_cat))