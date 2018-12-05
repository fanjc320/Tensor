

import numpy as np;
import codecs
import re
import math

a = 'hello word'
strinfo = re.compile('word')
b = strinfo.sub('python',a)
print(b)

# fin = open('tmp.csv','r',encoding='utf8')
# finLines =[];
# fout= open('E:/Tensor/LoggerTool_py/tmp_out.csv','w',encoding='utf8');
# count = 0;
# contents = [];

# for line in fin.readlines():
    # line = line.strip()
    # #finLines.append(line)
    # elements = line.split(',');
    # order = 0
    # for ele in elements:
        # if order == 0:
            # print("{uid="+ele+",",end=" ",file=fout)
        # if order == 1:
            # print("time=\""+ele+"\",",end=" ",file=fout)
        # if order == 2:
            # print("add_reduce="+ele+",",end=" ",file=fout)
        # if order == 3:
            # print("cfgid="+ele+",",end=" ",file=fout)
        # if order == 4:
            # print("mkid=\""+ele+"\"},",end=" ",file=fout)
        # order = order+1;
        # #print(elements,file=fout)
    # print("",file=fout)
		
# fin.close()

 
def HandleStringInColums(fin,fout):
    fin = open(fin,'r',encoding='utf8')
    fout= open(fout,'w',encoding='utf8');
    rowsCnt = 0;
    for line in fin.readlines():
        line = line.strip();
        rowsCnt = rowsCnt+1;
        column = 0;
        elements = line.split('	');
        for ele in elements:
            if column == 0:
                print("["+ele+"]=",end=" ",file=fout)
            if column == 1:
                print("\""+ele+"\",",end=" ",file=fout)
            column = column+1;
            #print(elements,file=fout)
    print("",file=fout)
    print("共有 %d 行"%(rowsCnt));
    fin.close()

HandleStringInColums('pet_mkid.txt','pet_mkid_out.txt');
HandleStringInColums('petegg_mkid.txt','petegg_mkid_out.txt');
# Python 2 默认以字节流（对应 Python 3 的 bytes）的方式读文件，不像 Python 3 默认解码为 unicode。
# 如果文件内容不是unicode编码的，要先以二进制方式打开，读入比特流，再解码。'rb'  #二进制方式打开


# 利用下面命令, 可以轻松吧制表符转换成为空格, MARK 一下. 
# sed -i 's/\t/  /g' *.py