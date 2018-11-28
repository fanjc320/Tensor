

import numpy as np;
import codecs
import re
import math

a = 'hello word'
strinfo = re.compile('word')
b = strinfo.sub('python',a)
print(b)

fin = open('duiying.csv','r',encoding='utf8')
finLines =[];
fout= open('duiying_out.csv','w',encoding='utf8');
count = 0;
contents = [];
mkid_add = [];
mkid_reduce = [];
for line in fin.readlines():
    line = line.strip()
    #finLines.append(line)
    elements = line.split(',');
    # uid =0;time="";add_reduce = 0;cfgid =0;mkid = 0;
    #print("elements:"+str(len(elements)))
    #elements = map(eval,elements)
    uid,time,add_reduce,cfgid,mkid=elements;
    if add_reduce == "0":
        mkid_add.append(elements)
    else:
        mkid_reduce.append(elements)

print(len(mkid_add) )
print(len(mkid_reduce) )

order = 0;
fin = open('old.txt','r',encoding='utf8')
for line in fin.readlines():
    line = line.strip()
    #finLines.append(line)
    elements = line.split(':');
    order = order+1;
    # uid =0;time="";add_reduce = 0;cfgid =0;mkid = 0;
    #print("elements:"+str(len(elements)))
    uidstr,uid,value,rate,mkid,add_reduce,str=elements;
    index = 0;duiying = 0;
    # print("--mkid: %s " %(mkid))
    try:
        # mkid = int(mkid)
        print("----mkid: %s " %(mkid))
        index = mkid_reduce.index(mkid)
        print(index)
        print("----index: %d " %(index))
        duiying = mkid_add[index];
    except:
        a = 1
        # print("except mkid: %d" % mkid);
    # print("duiying: %d" %(duiying));

fin.close()

list1 = []
# Python 2 默认以字节流（对应 Python 3 的 bytes）的方式读文件，不像 Python 3 默认解码为 unicode。
# 如果文件内容不是unicode编码的，要先以二进制方式打开，读入比特流，再解码。'rb'  #二进制方式打开


# 利用下面命令, 可以轻松吧制表符转换成为空格, MARK 一下. 
# sed -i 's/\t/  /g' *.py