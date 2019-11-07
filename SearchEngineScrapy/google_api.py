from googleapiclient.discovery import build
# my_api_key = "Your API Key”
# my_cse_id = "Your CSE ID"
import requests
import json
import csv



def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res


def search(query,api_key,cse_id,page):
    para={
        'key':api_key,
        'cx':cse_id,
        'q':query,
        'start':page*10+1
    }
    r=requests.get('https://www.googleapis.com/customsearch/v1',params=para)
    result_json=json.loads(r.text)
    return result_json
if __name__=='__main__':
    api_keys=[
        'AIzaSyAAGaqcZsqwq9j5POh_DJzV5nqwTFTE3KU',
        'AIzaSyC6oNMrj78g-aHAHsWthpAss44ANMnQq7c',
        'AIzaSyDum7f7tkl4LTqHgbVpNZ7Anzi2ksB1ibA',
        'AIzaSyAml4Gv59qPY4zT_fraDEY6bcDy4gCZ-pU',
        'AIzaSyDAHxazXAKTyplxHsZWWlNB9AdxNezyLCM',

        'AIzaSyD90sTjUN78CGWkkPnsXnNZoW9L4KjyrV0',
        'AIzaSyA2nbuPVsVWMNJzPqpTC84rnZcrjtWAQv8',
        'AIzaSyD347p7XvYk0xnXXdRBbylQN_itAx4Byes',
        'AIzaSyB17dVlruLqRRPpg6GUdaLE7Gom82FK3nI'
    ]
    my_api_key = api_keys[1]
    my_cse_id = '014747323697270263614:ogs0sc5goyq'
    result_file='../data/result_categories_retailer.csv'
    search_query_file='../data/search_query_categories.csv'


    query_list = []
    with open(search_query_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            query_list.append(line.strip() + ' retailer')
    # query_list=['food']
    with open(result_file, 'r',encoding='utf-8') as f:
        search_results = csv.reader(f, delimiter='\t')
        results = set([(line[0], line[1]) for line in search_results])

    file = open(result_file, 'a', encoding='utf-8')

    for count,searchQuery in enumerate(query_list):
        # searchQuery='fish reseller'
        searchQuery = searchQuery.lower()
        searchEngine = 'google'
        pages = 10

        if count>=10:
            break
        for page in range(pages):
            if (searchQuery, str(page)) in results:
                continue
            try:
                result = search(searchQuery, my_api_key, my_cse_id, page)
                search_items = [[i['title'].replace('\n', '').replace('\t', ''), i['link'],
                                 i['snippet'].replace('\n', '').replace('\t', '')] for i in result['items']]

            except Exception:
                print('error')
                continue
            # search_items=[[i['title'].replace('\n','').replace('\t',''),i['link'],i['snippet'].replace('\n','').replace('\t','')] for i in result['items']]


            print('finish page ',page)
            line = searchQuery + '\t' + str(page) + '\t' + str(search_items) + '\n'
            file.write(line)
        print('finish query ',count)
    file.close()

