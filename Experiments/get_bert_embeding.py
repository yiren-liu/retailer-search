from bert_serving.client import BertClient
from keras.preprocessing.text import Tokenizer
from copy import deepcopy
import ast
import csv
import json
import string
import re
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
def handle_search_result(search_result):
    title=search_result[0].translate(str.maketrans('', '',string.punctuation))
    description=search_result[2].translate(str.maketrans('', '',string.punctuation))
    #print(search_result[1])
    link=' '.join(re.split('\.|/|-|\?|=|&',search_result[1].split('//')[1]))
    #print(link)
    return title+' '+link+' '+description



def read_description_file():
    description_file = '../data/description_categories.csv'
    with open(description_file, 'r', encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        all_description={}
        all_title={}
        for line in search_results:
            try:
                all_description[(line[0], line[1],line[2])]=line[4].translate(str.maketrans('', '',string.punctuation))
                all_title[(line[0], line[1],line[2])]=line[3].translate(str.maketrans('', '',string.punctuation))
            except Exception:
                print('error')
                continue
    return all_title,all_description

def read_label_file():
    label_file = '../data/label_data_retailer_categories.csv'
    with open(label_file, 'r', encoding='utf-8') as f:
        labels={}
        search_results = csv.reader(f, delimiter='\t')
        for line in search_results:
            labels[(line[0], line[1],line[2])]=line[3]
    return labels

def read_result_file():
    result_file = '../data/result_categories_retailer.csv'
    with open(result_file, 'r', encoding='utf-8') as f:
        f = csv.reader(f, delimiter='\t')
        all_search_results={}
        for i, line in enumerate(f):

            one_page_results = ast.literal_eval(line[2])

            for count, one_result in enumerate(one_page_results):
                all_search_results[(line[0], line[1], str(count))] = handle_search_result(one_result)
    return all_search_results



def creat_2_cat_label(out_label):
    ult_label=[]
    for i in out_label:
        if i==0 or i==2:
            ult_label.append(0)
        else:
            ult_label.append(1)
    np.save('../data/labels_2_cat_big.npy', np.array(ult_label))

def creat_3_cat_label(out_label):
    np.save('../data/labels_3_cat_big.npy', out_label)

def get_batch_encoding(data,split_num):

    split=int(len(data)/split_num)
    out_data=[]
    for i in range(split_num):
        bc = BertClient()
        if i == split_num-1:
            temp=bc.encode(descriptions[i*split:])
        else:
            temp=bc.encode(descriptions[i*split:(i+1)*split])

        out_data.append(temp)
    maxlen=0
    for i in out_data:
        if len(i[0])>maxlen:
            maxlen=len(i[0])
    return maxlen

def clean_sentence(sentence):
    fil = re.compile(u'[^0-9a-zA-Z ]+', re.UNICODE)
    x=fil.sub('', sentence)

    #print(sentence)
    #print(x)
    return x

def stat_words(descriptions,results):
    descriptions_num=[]
    for sen in descriptions:
        descriptions_num.append(len(sen.split(' ')))

    results_num = []
    for sen in results:
        results_num.append(len(sen.split(' ')))

    import pickle
    with open('descriptions_results_num_params.sav', 'wb') as f:
        pickle.dump([descriptions_num,results_num], f, -1)

# def get_one_hot_dict(descriptions, results):
#     des_one_hot = []
#     res_one_hot = []
#     all_word_set = set([word for sentence in descriptions for word in sentence.split(" ")])
#     all_word_set.update([word for sentence in results for word in sentence.split(" ")])
#     one_hot_set = []
#     for idx, word in enumerate(all_word_set):
#         one_hot = [0 for _ in range(len(all_word_set))]
#         one_hot[idx] = 1
#         one_hot_set.append(one_hot)
#     return dict(zip(all_word_set,one_hot_set))


def create_bert_embeding(descriptions,results,out_label):

    # num=0
    # split_num=16
    # split = int(len(descriptions) / split_num)
    bc = BertClient()
    # for num in range(split_num):
    #     # if num<=1:
    #     #     continue
    #
    #     if num == split_num-1:#from 0 to 3
    #         description = bc.encode(descriptions[num * split:])
    #     else:
    #         description = bc.encode(descriptions[num * split:(num + 1) * split])
    #
    #     if num == split_num-1:#from 0 to 3
    #         result = bc.encode(results[num * split:])
    #     else:
    #         result = bc.encode(results[num * split:(num + 1) * split])


    descriptions.extend(results)
    # split=int(len(descriptions)/16)


    descriptions=bc.encode(descriptions)
    results=descriptions[len(results):]
    descriptions=descriptions[:len(results)]
    # descriptions_1=bc.encode(descriptions[:split])
    # results = bc.encode(results)
    # get_batch_encoding(descriptions,16)
    # get_batch_encoding(results,16)



    np.save('../data/descriptions_big.npy',descriptions)
    np.save('../data/results_big.npy', results)
    out_label=np.array(out_label)
    creat_2_cat_label(out_label)
    creat_3_cat_label(out_label)

    print('sample num: ',len(descriptions))
    print('max_seq_len: ',len(descriptions[0]))

    # print(bc.encode(['First do it', 'then do it right', 'then do it better']).shape)


def create_one_hot_embeding(descriptions,results,out_label):
    # get one-hot
    all_text = deepcopy(descriptions)
    all_text.extend(results)
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(all_text)

    descriptions = pad_sequences(tokenizer.texts_to_sequences(descriptions),maxlen=100)
    results = pad_sequences(tokenizer.texts_to_sequences(results),maxlen=100)

    np.save('../data/descriptions_big_one_hot_index.npy', descriptions)
    np.save('../data/results_big_one_hot_index.npy', results)




def create_ngram_list(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return list(zip(*[input_list[i:] for i in range(ngram_value)]))

def create_1_gram_embeding(descriptions,results,out_labe):


    # get one-hot
    all_text = deepcopy(descriptions)
    all_text.extend(results)
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(all_text)
    descriptions_seq = tokenizer.texts_to_sequences(descriptions)
    results_seq = tokenizer.texts_to_sequences(results)
    all_seq=deepcopy(descriptions_seq)
    all_seq.extend(results_seq)
    # one_hot_all = tokenier.texts_to_matrix(all_text)
    # get BoW encoding

    BoW_des = tokenizer.texts_to_matrix(descriptions,mode='count')
    BoW_res = tokenizer.texts_to_matrix(results,mode='count')


    np.save('../data/descriptions_big_BoW_1_gram.npy', BoW_des)
    np.save('../data/results_big_BoW_1_gram.npy', BoW_res)


def tuple_to_index(ngram_set, all_seq_gram):
    start_index = 0
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    # indice_token = {token_indice[k]: k for k in token_indice}
    all_seq_indice = []
    for x in all_seq_gram:
        sent=[]
        for i in x:
            sent.append(token_indice[i])
        all_seq_indice.append(sent)
    return all_seq_indice

def create_n_gram(descriptions,results,out_label,n):
    all_text = deepcopy(descriptions)
    all_text.extend(results)
    bigram_vectorizer = CountVectorizer(ngram_range=(n, n),
                            token_pattern = r'\b\w+\b', min_df = 1,max_features=500)
    all_vec=bigram_vectorizer.fit_transform(all_text).toarray()

    split=len(results)
    results=all_vec[split:]
    descriptions=all_vec[:split]
    np.save('../data/descriptions_big_BoW_%d_gram.npy'%n, descriptions)
    np.save('../data/results_big_BoW_%d_gram.npy'%n, results)


if __name__=='__main__':
    all_title, all_description = read_description_file()
    labels = read_label_file()
    all_search_results = read_result_file()

    descriptions=[]
    results=[]
    out_label=[]
    for key in labels.keys():
        if labels[key]=='3':
            continue

        if key not in all_title.keys():
            continue
        if all_title[key]+all_description[key]=='  ':
            continue

        desc=clean_sentence(all_title[key] + all_description[key])
        res=clean_sentence(all_search_results[key])
        if desc.strip()=='' or res.strip()=='':
            continue
        if labels[key]=='1':
            out_label.append(1)
        elif labels[key]=='0':
            out_label.append(0)
        elif labels[key]=='2':
            out_label.append(2)
        else:
            continue

        descriptions.append(all_title[key] + all_description[key])
        results.append(all_search_results[key])

    # stat_words(descriptions,results)
    d=list(zip(descriptions,results,out_label))
    random.seed(59)
    random.shuffle(d)
    descriptions, results, out_label=zip(*d)
    descriptions=list(descriptions)
    results=list(results)

    # create_one_hot_embeding(descriptions, results, out_label)
    # create_n_gram(descriptions, results, out_label, 1)
    # create_n_gram(descriptions, results, out_label, 2)
    # create_n_gram(descriptions, results, out_label, 3)

    #get one-hot
    all_text = deepcopy(descriptions)
    all_text.extend(results)
    tokenier = Tokenizer(num_words = 100)
    tokenier.fit_on_texts(all_text)
    #one_hot_all = tokenier.texts_to_matrix(all_text)

    max_len = 200
    des_one_hot = []
    for sentence in descriptions:
        temp = np.zeros([max_len, 100]).tolist()
        for idx, word in enumerate(sentence.split(" ")):
            if idx > max_len - 1:
                break
            temp[idx]= tokenier.texts_to_matrix([word])[0]
        des_one_hot.append(temp)

    res_one_hot = []
    for sentence in results:
        temp = np.zeros([max_len, 100]).tolist()
        for idx, word in enumerate(sentence.split(" ")):
            if idx > max_len - 1:
                break
            temp[idx]= tokenier.texts_to_matrix([word])[0]
        res_one_hot.append(temp)

    des_one_hot = np.array(des_one_hot)
    res_one_hot = np.array(res_one_hot)


    # des_one_hot = one_hot_all[:len(descriptions)]
    # res_one_hot = one_hot_all[len(descriptions):]

    np.save('../data/descriptions_big_one_hot.npy',des_one_hot)
    np.save('../data/results_big_one_hot.npy', res_one_hot)
    
    #create_bert_embeding(descriptions, results, out_label)