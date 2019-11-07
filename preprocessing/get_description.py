from googleapiclient.discovery import build
# my_api_key = "Your API Key”
# my_cse_id = "Your CSE ID"
import requests
import json
import csv
import re
from lxml import etree
import ast
if __name__=='__main__':
    result_file='../data/result_retailer.csv'
    description_file='../data/description.csv'

    with open(description_file, 'r', encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        results = set([(line[0], line[1],line[2]) for line in search_results])

    all_result = []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        f=csv.reader(f, delimiter='\t')
        for i,line in enumerate(f):
            if i<=100:
                continue
            one_page_results=ast.literal_eval(line[2])

            for count,one_result in enumerate(one_page_results):
                if (line[0],line[1],str(count)) in results:
                    continue
                if one_result in all_result:
                    continue

                temp_url=one_result[1].replace('//','t')
                if '/' not in temp_url:
                    urt=one_result[1]
                elif re.match('https',one_result[1])==None:
                    url=re.match('http://.*?(?=/)',one_result[1]).group()

                else:
                    url=re.match('https://.*?(?=/)', one_result[1]).group()
                try:
                    r=requests.get(url)
                    html=etree.HTML(r.text)
                    description=html.xpath('/html/head/meta[@name="description"]/@content')[0]
                except Exception:
                    print('error')
                    continue
                # print('\n'+line[0]+'\n'+one_result[0]+'\n'+one_result[1]+'\n'+one_result[2])
                # label=input('1 为零售商，0为其他 2为生产商，请标注：')
                file = open(description_file, 'a', encoding='utf-8')
                file.write(line[0]+'\t'+line[1]+'\t'+str(count)+'\t'+description.replace('\t',' ')+'\n')
                file.close()
                print('count ',count)
            print('page ',i)
