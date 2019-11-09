from scrapy.selector import Selector
from scrapy.spiders import Spider
import scrapy
from SearchEngineScrapy.items import descript_item
import re
from SearchEngineScrapy.utils.searchengines import SearchEngineResultSelector
from SearchEngineScrapy.utils.searchenginepages import SearchEngineURLs
import csv
from urllib.parse import urljoin
import ast
import random

class SearchEngineScrapy(Spider):
    name = "SearchEngineScrapy"

    # allowed_domains = ['bing.com','google.com']
    # start_urls = []

    searchQuery = None
    searchEngine = None
    selector = None

    def start_requests(self):

        # super(SearchEngineScrapy, self).__init__(*args, **kwargs)
        result_file = '../data/result_categories_retailer.csv'
        description_file = '../data/description_categories.csv'

        with open(description_file, 'r', encoding='utf-8') as f:
            search_results = csv.reader(f, delimiter='\t')
            results=set()
            for line in search_results:
                try:
                    results.add((line[0], line[1], line[2]))
                except Exception:
                    print('error')
                    continue

        all_result = []

        with open(result_file, 'r', encoding='utf-8') as f:
            f = csv.reader(f, delimiter='\t')
            f=list(f)
            random.shuffle(f)
            for i, line in enumerate(f):
                # if i <= 100:
                #     continue
                # if i>=102:
                #     continue
                one_page_results = ast.literal_eval(line[2])

                for count, one_result in enumerate(one_page_results):
                    if (line[0], line[1], str(count)) in results:
                        continue
                    if one_result in all_result:
                        continue

                    temp_url = one_result[1].replace('//', 't')
                    if '/' not in temp_url:
                        urt = one_result[1]
                    elif re.match('https', one_result[1]) == None:
                        url = re.match('http://.*?(?=/)', one_result[1]).group()

                    else:
                        url = re.match('https://.*?(?=/)', one_result[1]).group()
                    yield scrapy.Request(url=url, callback=self.parse, meta={'searchQuery': line[0],'page':line[1],'count':count})

    def parse(self, response):
        item = descript_item()


        description = response.xpath('/html/head/meta[@name="description"]/@content')
        if description==[]:
            description=response.xpath('/html/head/meta[@name="Description"]/@content')
        if description==[]:
            description=response.xpath('/html/head/meta[@name="DESCRIPTION"]/@content')


        if description==[]:
            description=response.xpath('/html/head/meta[@property="og:description"]/@content')
        if description==[]:
            description=response.xpath('/html/head/meta[@property="og:Description"]/@content')
        if description==[]:
            description=response.xpath('/html/head/meta[@property="og:DESCRIPTION"]/@content')

        if description!=[]:
            description=description.extract()[0].replace('\t', ' ').replace('\n', ' ').replace('\r',' ').replace('\r\n',' ')
        else:
            description=' '
            print('no description')

        try:
            title = response.xpath('/html/head/title/text()').extract()[0].replace('\t',' ').replace('\n', ' ').replace('\r',' ').replace('\r\n',' ')
        except Exception:
            print('no title')
            title=' '

        if description==' ' or description=='':
            return
        item['title']=title
        item['query']=response.meta['searchQuery']
        item['count'] = response.meta['count']
        item['page'] = response.meta['page']
        item['description']=description
        yield item

        # if self.count < 22:
        #     next_link = response.xpath('//a[@id="pnnext"]/@href').extract()
        #     if next_link:
        #         next_link = urljoin(response.url, next_link[0])
        #         self.count += 1
        #         yield scrapy.Request(url=next_link, headers=self.headers, callback=self.parse)