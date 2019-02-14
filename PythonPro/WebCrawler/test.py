import requests
import urllib
import re
import random
from time import sleep
def main():
    ulr = '知乎-与世界分享你的知识、经验和见解'

    headers = {省略}
    i=1
    for x in xrange(20,3600,20):
        data = {'start':'0',
                'offset':str(x),
                '_xsrf':'kkkk'}

    content = requests.post(url,headers=headers,data=data,timeout=10).text
    print("0000:",content);
    imgs = re.findall('<img src =\\\\"(.*?)_m.jpg',content)
    print("imgs",imgs)
    for img in imgs:
        try:
            print("img:"+img)
            #img=img.replace('\\',")
            pic =img+'.jpg'
            path='d:\\bs4\\zhihu\\jpg\\'+str(i)+'.jpg'

            urllib.urlretrieve(pic,path)
            print('下载了第'+str(i)+u'张图片')
            i+=1

            sleep(random.uniform(0.5,1))
        except:
            print("louuuu")
            pass
            sleep(random.uniform(0.5,1))

if __name__=='__main__'

main()