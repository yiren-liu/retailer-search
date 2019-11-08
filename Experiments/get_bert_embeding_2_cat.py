from bert_serving.client import BertClient
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
    label_file = '../data/label_data_retailer_categories_rem_dup.csv'
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
        if all_title[key]+all_description[key]=='  ':
            continue
        descriptions.append(all_title[key]+all_description[key])
        results.append(all_search_results[key])
        if int(labels[key])==1:
            out_label.append(1)
        else:
            out_label.append(-1)


    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(descriptions)
    random.seed(randnum)
    random.shuffle(results)
    random.seed(randnum)
    random.shuffle(out_label)

    bc = BertClient()
    descriptions.extend(results)

    all=bc.encode(descriptions)
    descriptions=all[:len(results)]
    results = all[len(results):]
    np.save('../data/descriptions.npy',descriptions)
    np.save('../data/results.npy', results)
    out_label=np.array(out_label)
    np.save('../data/labels_2_cat.npy', out_label)



    print(bc.encode(['First do it', 'then do it right', 'then do it better']).shape)

