import json
import requests
from requests.exceptions import RequestException
import re
import time
from urllib import request
import os

def get_one_page(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # print("response.text:"+str(response.text)+"-----\n")
            return response.text
        print("response.status_code:"+str(response.status_code))
        return None
    except RequestException:
        return None

def parse_one_page(html):
    pattern = re.compile('<dd>.*?board-index.*?>(\d+)</i>.*?data-src="(.*?)".*?name"><a'
                         + '.*?>(.*?)</a>.*?star">(.*?)</p>.*?releasetime">(.*?)</p>'
                         + '.*?integer">(.*?)</i>.*?fraction">(.*?)</i>.*?</dd>', re.S)
    items = re.findall(pattern, html)
    for item in items:
        yield {
            'index': item[0],
            'image': item[1],
            'title': item[2],
            'actor': item[3].strip()[3:],
            'time': item[4].strip()[5:],
            'score': item[5] + item[6]
        }
		
def parse_one_page_My(html):
    # print("html:"+html+"=====\n")
    # pattern = re.compile('<script .*?(.*?).*?</script>', re.S)
    pattern = re.compile('<script.*?src=\'(.*?)\'></script>', re.S)
    items = re.findall(pattern, html)
    # items = pattern.findall(html)
    # print(items)
    for item in items:
        # print("item:"+item)
        yield item

def write_to_file(content):
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False) + '\n')

def main(offset):
    # url = 'http://maoyan.com/board/4?offset=' + str(offset)
    url = 'https://bideyuanli.com/pp'
    html = get_one_page(url)
    #for item in parse_one_page(html):
    for item in parse_one_page_My(html):
        print(item)
        filename = os.path.basename(item)
        # 去掉？
        filename = re.split(r'\?',filename)[0]
        print(filename)
        request.urlretrieve(item,"E:/Tensor/PythonPro/WebCrawler_1/bideyuanli/"+filename)
        write_to_file(item)

if __name__ == '__main__':
    main(0)
    # for i in range(10):
        # main(offset=i * 10)
        # time.sleep(1)