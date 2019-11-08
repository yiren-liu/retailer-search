import ast
import csv
import json
from preprocessing_funs import *
def tag_data():
    result_file='../data/result_categories_retailer.csv'
    label_file='../data/label_data_retailer_categories.csv'
    description_file = '../data/description_categories.csv'

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



    with open(label_file, 'r', encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        results = set([(line[0], line[1],line[2]) for line in search_results])

    all_result=[]
    # file= open('label_excel.csv','w',encoding='utf-8')
    with open(result_file, 'r', encoding='utf-8') as f:
        f=csv.reader(f, delimiter='\t')
        for i,line in enumerate(f):

            one_page_results=ast.literal_eval(line[2])

            for count,one_result in enumerate(one_page_results):
                if (line[0],line[1],str(count)) in results:
                    continue
                if one_result in all_result:
                    continue

                if (line[0],line[1],str(count)) not in all_description:#543开始
                    continue

                print('\n'+line[0]+'\n'+one_result[0]+'\n'+one_result[1]+'\n'+one_result[2])

                print('标题：',all_title[(line[0],line[1],str(count))])
                print('描述：',all_description[(line[0],line[1],str(count))])


                # file.write(line[0]+'\t'+one_result[0]+'\t'+one_result[1]+'\t'
                #            +one_result[2]+'\t'+all_title[(line[0],line[1],str(count))]
                #                                           +'\t'+all_description[(line[0],line[1],str(count))]+'\n')

                label=input('1 为零售商，0为其他 2为生产商，3为无法判断或网站描述缺失或描述没有作用为，请标注：')
                file = open(label_file, 'a', encoding='utf-8')
                file.write(line[0]+'\t'+line[1]+'\t'+str(count)+'\t'+str(label)+'\n')
                file.close()

            all_result.extend(one_page_results)
    # file.close()



def check_label():
    all_title, all_description=read_description_file()
    labels=read_label_file()
    all_search_results=read_result_file()
    pass
    for i in labels.keys():
        if i in all_description.keys():
            if all_description[i]!=' ':
                print('\n'+str(i)+'\n'+all_search_results[i][0]+'\n'+all_search_results[i][1]+'\n'+all_search_results[i][2])
                print('标题：', all_title[i])
                print('描述：', all_description[i])
                print('label: ',labels[i])
                print('\n')


if __name__=='__main__':
    tag_data()
