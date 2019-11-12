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

def handle_search_result(search_result):
    title=search_result[0].translate(str.maketrans('', '',string.punctuation))
    description=search_result[2].translate(str.maketrans('', '',string.punctuation))
    print(search_result[1])
    link=' '.join(re.split('\.|/|-|\?|=|&',search_result[1].split('//')[1]))
    print(link)
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

    print(sentence)
    print(x)
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
    random.seed(34)
    random.shuffle(d)
    descriptions, results, out_label=zip(*d)
    descriptions=list(descriptions)
    results=list(results)

    #get one-hot
    all_text = deepcopy(descriptions)
    all_text.extend(results)
    tokenier = Tokenizer(num_words = 1000)
    tokenier.fit_on_texts(all_text)
    #one_hot_all = tokenier.texts_to_matrix(all_text)

    max_len = 200
    des_one_hot = []
    for sentence in descriptions:
        temp = np.zeros([max_len, 1000]).tolist()
        for idx, word in enumerate(sentence.split(" ")):
            temp[idx]= tokenier.texts_to_matrix([word])[0]
        des_one_hot.append(temp)

    res_one_hot = []
    for sentence in results:
        temp = np.zeros([max_len, 1000]).tolist()
        for idx, word in enumerate(sentence.split(" ")):
            temp[idx]= tokenier.texts_to_matrix([word])[0]
        res_one_hot.append(temp)

    des_one_hot = np.array(des_one_hot)
    res_one_hot = np.array(res_one_hot)


    # des_one_hot = one_hot_all[:len(descriptions)]
    # res_one_hot = one_hot_all[len(descriptions):]

    np.save('../data/descriptions_big_one_hot.npy',des_one_hot)
    np.save('../data/results_big_one_hot.npy', res_one_hot)

    # max_len = 200
    # one_hot_dict = get_one_hot_dict(descriptions, results)

    # des_one_hot = []
    # for sentence in descriptions:
    #     temp = np.zeros([max_len, len(one_hot_dict)]).tolist()
    #     for idx, word in enumerate(sentence.split(" ")):
    #         temp[idx]= one_hot_dict[word]
    #     des_one_hot.append(temp)

    # res_one_hot = []
    # for sentence in results:
    #     temp = np.zeros([max_len, len(one_hot_dict)]).tolist()
    #     for idx, word in enumerate(sentence.split(" ")):
    #         temp[idx]= one_hot_dict[word]
    #     res_one_hot.append(temp)

    # des_one_hot = np.array(des_one_hot)
    # res_one_hot = np.array(res_one_hot)

    # np.save('../data/descriptions_big_one_hot.npy',des_one_hot)
    # np.save('../data/results_big_one_hot.npy', res_one_hot)


    num=0
    split_num=16
    split = int(len(descriptions) / split_num)
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
    print('num %d finished'%num)
    # print(bc.encode(['First do it', 'then do it right', 'then do it better']).shape)

