import ast
import csv
import json
import string
import collections
import re
import os
from preprocessing_funs import *
import pickle
import matplotlib.pyplot as plt

import seaborn as sns
def handle_search_result(search_result):
    title=search_result[0].translate(str.maketrans('', '',string.punctuation))
    description=search_result[2].translate(str.maketrans('', '',string.punctuation))
    print(search_result[1])
    link=' '.join(re.split('\.|/|-|\?|=|&',search_result[1].split('//')[1]))
    print(link)
    return title+' '+link+' '+description

def creat_model_input_data():
    result_file = '../data/result_retailer.csv'
    label_file = '../data/label_data_retailer.csv'

    all_search_results={}
    with open(result_file, 'r', encoding='utf-8') as f:
        f = csv.reader(f, delimiter='\t')
        for i, line in enumerate(f):

            one_page_results = ast.literal_eval(line[2])

            for count, one_result in enumerate(one_page_results):
                all_search_results[(line[0],line[1],str(count))]=one_result


    labels=[]
    handled_results=[]
    with open(label_file, 'r', encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        for line in search_results:
            one_search_result=all_search_results[(line[0], line[1],line[2])]
            handled_result=handle_search_result(one_search_result)
            handled_results.append(handled_result)

            labels.append(line[3])
    print(len(handled_results))
    split=int(len(handled_results)*9/10)
    with open('../data/train.tsv','w',encoding='utf-8') as f:
        for i,x in enumerate(handled_results[:split]):
            f.write(x+'\t'+labels[i]+'\n')
    with open('../data/dev.tsv','w',encoding='utf-8') as f:
        for i,x in enumerate(handled_results[split:]):
            f.write(x+'\t'+labels[i+split]+'\n')

def data_stats():
    result_file = '../data/result_retailer.csv'
    label_file = '../data/label_data_retailer_categories_rem_dup.csv'

    # all_search_results = {}
    # with open(result_file, 'r', encoding='utf-8') as f:
    #     f = csv.reader(f, delimiter='\t')
    #     for i, line in enumerate(f):
    #
    #         one_page_results = ast.literal_eval(line[2])
    #
    #         for count, one_result in enumerate(one_page_results):
    #             all_search_results[(line[0], line[1], str(count))] = one_result


    labels = []
    handled_results = []

    with open(label_file, 'r', encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        for line in search_results:
            # one_search_result=all_search_results[(line[0], line[1],line[2])]
            # handled_result=handle_search_result(one_search_result)
            # handled_results.append(one_search_result)

            labels.append(line[3])

    dic = collections.Counter(labels)

    for key in dic:
        print(key, dic[key])
    # a=[]
    # b=[]
    # c=[]
    # for i,x in enumerate(labels):
    #     if x=='0':
    #         a.append(handled_results[i])
    #     elif x=='1':
    #         b.append(handled_results[i])
    #     else:
    #         c.append(handled_results[i])
    #
    # for i,x in enumerate([a,b,c]):
    #     with open('../data/label_data/label_%s_data.csv'%str(i),'w',encoding='utf-8') as f:
    #         for m in x:
    #             f.write(str(m)+'\n')



    pass

#对search query处理，去掉其中重复的行
def remove_dup_query(file):
    with open(file,'r') as f:
        lines_set=set()
        lines=f.readlines()
        nums=[]
        for i,line in enumerate(lines):
            if line.strip().lower() not in lines_set:
                nums.append(i)
                lines_set.add(line.strip().lower())
    if os.path.exists(file + 'bac'):
        os.remove(file + 'bac')
    os.rename(file, file + 'bac')
    with open(file,'w') as f:
        for x in lines_set:
            f.write(x+'\n')

#对descriptions进行处理，移除描述为空的网站
def remove_dup_description(file):
    all_title,all_descr=read_description_file(file)
    if os.path.exists(file + 'bac'):
        os.remove(file + 'bac')
    os.rename(file,file+'bac')
    f=open(file,'w',encoding='utf-8')
    for key in all_descr.keys():
        if all_descr[key]==' 'or all_descr[key]=='':
            continue
        f.write(key[0]+'\t'+key[1]+'\t'+key[2]+'\t'+all_title[key]+'\t'+all_descr[key]+'\n')
    f.close()


def replace_npy():
    import numpy as np
    a=np.load('../data/labels_2_cat.npy')
    b=[]
    for i in a:
        if i==0 or i==2:
            b.append(0)
        elif i==1:
            b.append(1)
    np.save('../data/labels_2_cat.npy',np.array(b))


def rem_dup():
    labels=read_label_file()
    pass
    with open('../data/label_data_retailer_categories_rem_dup.csv','w') as f:
        for key in labels.keys():
            f.write(key[0]+'\t'+key[1]+'\t'+key[2]+'\t'+labels[key]+'\n')

def draw_pic():

    with open('history_params.sav', 'rb') as f:
        tmp = pickle.load(f)
    history_dict = tmp
    loss_values = history_dict['regression_output_loss']
    val_loss_values = history_dict['val_regression_output_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')  # ←------'bo'
    # 表示蓝色圆点
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # ←------'b'
    # 表示蓝色实线
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    plt.clf()  # ←------ 清空图像
    acc = history_dict['regression_output_categorical_accuracy']
    val_acc = history_dict['val_regression_output_categorical_accuracy']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def draw_word_num_pic():

    with open('descriptions_results_num_params.sav', 'rb') as f:
        tmp = pickle.load(f)
    des_num=tmp[0]
    resu_num=tmp[1]

    # sns.kdeplot(des_num,shade=True)#bins=60,histtype="stepfilled", alpha=.8
    # plt.title('Words number distribution of website desriptions')
    # plt.xlabel('Number of words ')
    # plt.ylabel('Probability')
    plt.show()
    plt.clf()

    # sns.kdeplot(resu_num,shade=True)
    # plt.title('Words number distribution of search results')
    # plt.xlabel('Number of words ')
    # plt.ylabel('Probability')
    plt.show()
    print(len(des_num))

if __name__=='__main__':
    # remove_dup_query('../data/search_query_categories.csv')
    pass
    draw_pic()
    # replace_npy()
    # remove_dup_description('../data/description_categories.csv')
    # draw_word_num_pic()