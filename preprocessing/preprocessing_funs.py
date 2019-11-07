def read_description_file():
    description_file = '../data/description.csv'
    with open(description_file, 'r', encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        all_description={}
        all_title={}
        for line in search_results:
            try:
                all_description[(line[0], line[1],line[2])]=line[4]
                all_title[(line[0], line[1],line[2])]=line[3]
            except Exception:
                print('error')
                continue
    return all_title,all_description


import ast
import csv
import json


def read_label_file():
    label_file = '../data/label_data_retailer_categories.csv'
    with open(label_file, 'r', encoding='utf-8') as f:
        labels={}
        search_results = csv.reader(f, delimiter='\t')
        for line in search_results:
            # if (line[0], line[1],line[2]) in labels.keys():
            #     continue
            labels[(line[0], line[1],line[2])]=line[3]
    return labels

def read_result_file():
    result_file = '../data/result_retailer.csv'
    with open(result_file, 'r', encoding='utf-8') as f:
        f = csv.reader(f, delimiter='\t')
        all_search_results={}
        for i, line in enumerate(f):

            one_page_results = ast.literal_eval(line[2])

            for count, one_result in enumerate(one_page_results):
                all_search_results[(line[0], line[1], str(count))] = one_result
    return all_search_results