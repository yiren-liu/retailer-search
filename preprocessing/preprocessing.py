import ast
import csv
import json
import string
import collections
import re
import os
from preprocessing_funs import *
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
    os.rename(file,file+'bac')
    f=open(file,'w',encoding='utf-8')
    for key in all_descr.keys():
        if all_descr[key]==' ':
            continue
        f.write(key[0]+'\t'+key[1]+'\t'+key[2]+'\t'+all_descr[key]+'\t'+all_descr[key]+'\n')
    f.close()







def rem_dup():
    labels=read_label_file()
    pass
    with open('../data/label_data_retailer_categories_rem_dup.csv','w') as f:
        for key in labels.keys():
            f.write(key[0]+'\t'+key[1]+'\t'+key[2]+'\t'+labels[key]+'\n')

if __name__=='__main__':
    remove_dup_query('../data/search_query_categories.csv')
    pass
    # remove_dup_description('../data/description_categories.csv')