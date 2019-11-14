import pickle
from collections import defaultdict
from keras.utils import to_categorical
import numpy as np
import scipy.io as sio
import random
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# GLOBAL Vars
data_path = '../data/'
dataset_path = data_path + 'mosi/'
truth_path = dataset_path + 'Meta_data/boundaries_sentimentint_avg.csv'

# features
openface_path = dataset_path + "Features/Visual/OpenfaceRaw/"
facet_path = dataset_path + "Features/Visual/FACET_GIOTA/"
covarep_path = dataset_path + "Features/Audio/raw/"
transcript_path = dataset_path + 'Transcript/SEGMENT_ALIGNED/'

# embeddings
word2ix_path = data_path + 'glove_word_embedding/word2ix_300_mosi.pkl'
word_embedding_path = data_path + "glove_word_embedding/glove_300_mosi.pkl"


def load_word_embedding():
  with open(word_embedding_path) as f:
    return pickle.load(f)


def load_word2ix():
  with open(word2ix_path) as f:
    word2ix = pickle.load(f)
  return word2ix


# load meta data truth_dict[video_id][seg_id]
def load_truth():
  truth_dict = defaultdict(dict)
  with open(truth_path) as f:
    lines = f.read().split("\r\n")
  for line in lines:
    if line != '':
      line = line.split(",")
      truth_dict[line[2]][line[3]] = {'start_time': float(line[0]),
                                      'end_time': float(line[1]),
                                      'sentiment': float(line[4])}
  return truth_dict


def load_facet(truth_dict):
  for video_index in truth_dict:
    file_name = facet_path + video_index + '.FACET_out.csv'
    # print file_name
    with open(file_name) as f:
      lines = f.read().split('\r\n')[1:]
      lines = [[float(x) for x in line.split(',')] for line in lines if
               line != '']
      for seg_index in truth_dict[video_index]:
        for w in truth_dict[video_index][seg_index]['data']:
          start_frame = int(w['start_time_clip'] * 30)
          end_frame = int(w['end_time_clip'] * 30)
          ft = [line[5:] for line in lines[start_frame:end_frame]]
          if ft == []:
            avg_ft = np.zeros(len(lines[0]) - 5)
          else:
            # print np.array(ft).shape
            # print ft[0]
            avg_ft = np.mean(ft, 0)
          w['facet'] = avg_ft


def load_covarep(truth_dict):
  for video_index in truth_dict:
    file_name = covarep_path + video_index + '.mat'
    fts = sio.loadmat(file_name)['features']
    # print fts.shape
    for seg_index in truth_dict[video_index]:
      for w in truth_dict[video_index][seg_index]['data']:
        start_frame = int(w['start_time_clip'] * 100)
        end_frame = int(w['end_time_clip'] * 100)
        ft = fts[start_frame:end_frame]
        if ft.shape[0] == 0:
          avg_ft = np.zeros(ft.shape[1])
        else:
          # print np.array(ft).shape
          # print ft[0]
          avg_ft = np.mean(ft, 0)
        avg_ft[np.isnan(avg_ft)] = 0
        avg_ft[np.isneginf(avg_ft)] = 0
        w['covarep'] = avg_ft


def load_transcript(truth_dict, word2ix):
  for video_index in truth_dict:
    for seg_index in truth_dict[video_index]:
      file_name = transcript_path + video_index + '_' + seg_index
      truth_dict[video_index][seg_index]['data'] = []
      with open(file_name) as f:
        lines = f.read().split("\n")
        for line in lines:
          if line == '':
            continue
          line = line.split(',')
          # print line
          truth_dict[video_index][seg_index]['data'].append(
            {'word_ix': word2ix[line[1]], 'word': line[1],
             'start_time_seg': float(line[2]), 'end_time_seg': float(line[3]),
             'start_time_clip': float(line[4]),
             'end_time_clip': float(line[5])})


def split_data(tr_proportion, truth_dict):
  data = [(vid, truth_dict[vid]) for vid in truth_dict]
  data.sort(key=lambda x: x[0])
  tr_split = int(round(len(data) * tr_proportion))
  train = data[:tr_split]
  test = data[tr_split:]
  return train, test


def get_data(dataset, max_segment_len):
  print("Loading word embedding")
  eb = load_word_embedding()

  data = {'facet': [], 'covarep': [], 'text': [],
          'text_eb': [], 'label': [], 'id': []}
  for i in range(len(dataset)):
    # print dataset[i][0]
    v = dataset[i][1]
    for seg_id in v:
      fts = v[seg_id]['data']
      facet, text, covarep = [], [], []
      text_eb = []
      for w in fts[:max_segment_len]:
        text.append(w['word_ix'])
        text_eb.append(eb[w['word_ix']])
        covarep.append(w['covarep'])
        facet.append(w['facet'])
      for j in range(max_segment_len - len(text)):
        text.append(0)
        text_eb.append(eb[0])
        covarep.append(np.zeros(len(covarep[0])))
        facet.append(np.zeros(len(facet[0])))
      data['facet'].append(facet)
      data['covarep'].append(covarep)
      data['text'].append(text)
      data['text_eb'].append(text_eb)
      data['label'].append(v[seg_id]['sentiment'])
      data['id'].append(dataset[i][0] + '_' + seg_id)
  data['facet'] = np.array(data['facet'])
  data['covarep'] = np.array(data['covarep'])
  data['text'] = np.array(data['text'])
  data['text_eb'] = np.array(data['text_eb'])
  data['label'] = np.array(data['label'])
  return data


def load_word_level_features(max_seq_len, train_split):
  word2ix = load_word2ix()
  truth_dict = load_truth()
  load_transcript(truth_dict, word2ix)
  load_facet(truth_dict)
  load_covarep(truth_dict)
  train, test = split_data(train_split, truth_dict)
  train = get_data(train, max_seq_len)
  test = get_data(test, max_seq_len)
  return train, test

def load_search_data():
  data_1 = np.load('../../data/results_big.npy')
  data_2=np.load('../../data/descriptions_big.npy')
  labels_all=to_categorical(np.load('../../data/labels_3_cat_big.npy'))
  split = int(len(data_1) * 9 / 10)
  # facet_train = facet_all[0:3000]
  data_1_train = data_1[0:split]
  data_2_train = data_2[0:split]

  # facet_test = facet_all[3000:]
  data_1_test = data_1[split:]
  data_2_test = data_2[split:]

  y_train = labels_all[:split]
  y_test = labels_all[split:]

  feats_dict = {'t': [data_1_train, data_1_test, 'text'],
                # 'f': [facet_train, facet_test, 'video'],
                'c': [data_2_train, data_2_test, 'audio'],
                'train_labels': y_train,
                'test_labels': y_test
                }

  return feats_dict


def load_and_preprocess_data(max_seq_len=20,
                             train_split=2.0 / 3,
                             is_cycled=True):
  """
  API for loading data from disk for 3 modalities 

  Args:
    max_seq_len: discard the rest of the sentence if longer than this 
    train_split: the rest is test set (validation will be split from train
    is_cycled: cycle loss or not (for padding)

  Returns: features dict including each modality with train and set parts 

  """
  print("Loading data from disk...\n")
  # train, test = load_word_level_features(max_seq_len,
  #                                        train_split)

  # extract parts from datasets
  # facet_train = train['facet']
  # covarep_train = train['covarep'][:, :, 1:35]
  # facet_test = test['facet']
  # covarep_test = test['covarep'][:, :, 1:35]
  #
  # text_train = train['text_eb']
  # text_test = test['text_eb']
  # y_train = train['label']
  # y_test = test['label']

  # 随机生成（T,A,V）
  # facet_train = np.random.rand(100,50,20)
  # covarep_train = np.random.rand(100,50,20)
  # facet_test = np.random.rand(10,50,20)
  # covarep_test = np.random.rand(10,50,20)
  #
  # text_train = np.random.rand(100,50,30)
  # text_test = np.random.rand(10,50,30)
  # y_train = np.random.randint(0,2,100)
  # y_test = np.random.randint(0,2,10)

  #读取json
  facet_all=[]
  covarep_all=[]
  text_all=[]

  facet_train = []
  covarep_train = []
  facet_test = []
  covarep_test = []

  text_train = []
  text_test = []
  y_train = []
  y_test = []
  import json
  # with open('data.txt', 'r') as f:
  #   for i,line in enumerate(f.readlines()):
  #     data=json.loads(line)
  #
  #     text_all.append(data['SentSeq'])
  #     covarep_all.append(data['SpeedSeq'])
  #     facet_all.append([a + b for (a, b) in zip(data['VF_left'], data['VF_right'])])
  #
  #     pass

  x1_data = np.load("../data/results.npy")
  x2_data = np.load("../data/descriptions.npy")
  y_data = np.load("../data/labels_3_cat.npy") + 1

  x1_train, x1_test,x2_train, x2_test, y_train, y_test = train_test_split(x1_data,x2_data, y_data, train_size=0.8)
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)


  text_all = np.load('../data/results.npy')
  covarep_all=np.load('../data/descriptions.npy')
  labels_all=np.load('../data/labels.npy')

  # def padding(padding_list):
  #   max_len=max([len(x) for x in padding_list])
  #   padded_list=[]
  #   for i in padding_list:
  #     if len(i)<max_len:
  #       i.extend([0]*(max_len-len(i)))
  #     padded_list.append(i)
  #
  #   return padded_list
  #
  # def normalization(data):
  #   _range = np.max(data) - np.min(data)
  #   return (data - np.min(data)) / _range

  # facet_all=normalization(np.array(padding(facet_all)))
  # covarep_all = normalization(np.array(padding(covarep_all)))
  # text_all = normalization(np.array(padding(text_all)))

  # facet_all=facet_all.reshape(facet_all.shape[0], facet_all.shape[1], 1)
  # covarep_all = covarep_all.reshape(covarep_all.shape[0], covarep_all.shape[1], 1)
  # text_all = text_all.reshape(text_all.shape[0], text_all.shape[1], 1)

  # split = int(len(text_all) * 4 / 5)
  # # facet_train = facet_all[0:3000]
  # covarep_train = covarep_all[0:split]
  # text_train = text_all[0:split]
  #
  # # facet_test = facet_all[3000:]
  # covarep_test = covarep_all[split:]
  # text_test = text_all[split:]
  #
  #
  # y_train = labels_all[:split]
  # y_test = labels_all[split:]




  # --------------------------------------------
  #               INSTRUCTION
  # --------------------------------------------
  # Your data (T, A, V) each should be a 3D array
  # with size of:
  #   N x L x D_i
  # where
  #   N is number of samples
  #   L is the sentence / segment length
  #   D_i is the number of features (varies by T, A, V).
  # We do word-level aligned feature extraction
  # --------------------------------------------
  #
  # print("Post-processing facial and covarap features")
  # facet_train_max = np.max(np.max(np.abs(facet_train), axis=0), axis=0)
  # facet_train_max[facet_train_max == 0] = 1
  # facet_train = facet_train / facet_train_max
  # facet_test = facet_test / facet_train_max

  # text_dim = text_train.shape[2]
  # # facet_dim = facet_train.shape[2]
  # covarep_dim = covarep_train.shape[2]
  # max_dim = max(text_dim, covarep_dim)
  #
  # if is_cycled:
  #   print("PADDING FOR CYCLIC LOSS ...")
  #   if max_dim > text_dim:
  #     text_train = np.pad(text_train,
  #                         ((0, 0), (0, 0), (0, max_dim - text_dim)),
  #                         'constant')
  #     text_test = np.pad(text_test,
  #                        ((0, 0), (0, 0), (0, max_dim - text_dim)),
  #                        'constant')
  #   # if max_dim > facet_dim:
  #   #   facet_train = np.pad(facet_train,
  #   #                        ((0, 0), (0, 0), (0, max_dim - facet_dim)),
  #   #                        'constant')
  #   #   facet_test = np.pad(facet_test,
  #   #                       ((0, 0), (0, 0), (0, max_dim - facet_dim)),
  #   #                       'constant')
  #   if max_dim > covarep_dim:
  #     covarep_train = np.pad(covarep_train,
  #                            ((0, 0), (0, 0), (0, max_dim - covarep_dim)),
  #                            'constant')
  #     covarep_test = np.pad(covarep_test,
  #                           ((0, 0), (0, 0), (0, max_dim - covarep_dim)),
  #                           'constant')
  #
  # print("Text train: {}".format(text_train.shape))
  # print("Covarep train: {}".format(covarep_train.shape))
  # # print("Facet train: {}".format(facet_train.shape))

  feats_dict = {'t': [x1_train, x1_test, 'text'],
                # 'f': [facet_train, facet_test, 'video'],
                'c': [x2_train, x2_test, 'audio'],
                'train_labels': y_train,
                'test_labels': y_test
                }

  return feats_dict


if __name__ == "__main__":
  # we = load_word_embedding()
  tr_split = 2.0 / 3
  max_segment_len = 115
  train, test = load_word_level_features(max_segment_len,
                                         tr_split)
