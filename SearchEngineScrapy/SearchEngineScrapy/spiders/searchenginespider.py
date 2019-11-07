from scrapy.selector import Selector
from scrapy.spiders import Spider
import scrapy
from SearchEngineScrapy.items import ScrapyGooglesearchItem
import re
from SearchEngineScrapy.utils.searchengines import SearchEngineResultSelector
from SearchEngineScrapy.utils.searchenginepages import SearchEngineURLs
import csv
from urllib.parse import urljoin
class SearchEngineScrapy(Spider):
    name = "SearchEngineScrapy"

    allowed_domains = ['bing.com','google.com']
    start_urls = []

    searchQuery = None
    searchEngine = None
    selector = None

    def start_requests(self):

        # super(SearchEngineScrapy, self).__init__(*args, **kwargs)
        query_list=[]
        with open('../data/search_query.csv','r') as f:
            lines=f.readlines()
            for line in lines:
                query_list.append(line.strip()+' reseller')

        with open('../data/result.csv','r') as f:
            search_results=csv.reader(f,delimiter='\t')
            results = set([(line[0], line[1]) for line in search_results])


        for searchQuery in query_list:
            # searchQuery='fish reseller'
            searchQuery = searchQuery.lower()
            searchEngine = 'google'
            pages = 10

            pageUrls = SearchEngineURLs(searchQuery, searchEngine, pages)
            selector = SearchEngineResultSelector[searchEngine]

            for url,page in pageUrls:
                if (searchQuery, str(page)) in results:
                    continue
                yield scrapy.Request(url=url, callback=self.parse, meta={'searchQuery': searchQuery,'page':page})

    def parse(self, response):
        item = ScrapyGooglesearchItem()

        selector_array = response.xpath('//div[@class="g"]//div[@class="rc"]')
        # if self.count == 22:
        #     selector_array = selector_array[:5]
        all_results=[]
        if selector_array==[]:
            return

        for selector in selector_array:
            text = selector.xpath('.//h3/text()').extract()
            # name = re.search('(.*?) -', text[0], re.DOTALL)
            # item['name'] = name.group(1) if name else None
            # item['title'] = text[0].replace('\t',' ') if text else None
            link = selector.xpath('.//cite/text()').extract()[0]
            link=link.replace(' › ','/')
            # item['link'] = link if link else None
            # if '-' in text[0]:
            #     item['role'] = text[0].split('-')[1].strip()
            desc = selector.xpath('./div[@class="s"]//span[@class="st"]//text()').extract()
            # item['description'] = ''.join(desc).replace('\t', ' ') if desc else None

            all_results.append([text[0].replace('\t',' ') if text else None,link if link else None,''.join(desc).replace('\t', ' ') if desc else None])

        item['results']=all_results
        item['item'] = response.meta['searchQuery']
        item['page'] = response.meta['page']
        yield item

        # if self.count < 22:
        #     next_link = response.xpath('//a[@id="pnnext"]/@href').extract()
        #     if next_link:
        #         next_link = urljoin(response.url, next_link[0])
        #         self.count += 1
        #         yield scrapy.Request(url=next_link, headers=self.headers, callback=self.parse)