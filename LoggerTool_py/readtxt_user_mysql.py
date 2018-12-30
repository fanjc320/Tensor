

import numpy as np;
import codecs
import re
import math

a = 'hello word'
strinfo = re.compile('word')
b = strinfo.sub('python',a)
print(b)

#各行item个数不同，这两个不能用，genfromtxt可以指定列名
#a= np.loadtxt(fname = "./game.log",encoding='utf8',delimiter = '|',dtype = np.str)
#c= np.genfromtxt(fname = "./game.log",encoding='utf8',delimiter = '|',dtype = np.str)
 

# fo = open('./game.log','r',encoding='utf8')

# contents = [];
# for line in fo.readlines():
    # line = line.strip()
    # #print(line)
	
# fo.close()


fin = open('test1.txt','r',encoding='utf8')
finLines =[];
fout= open('E:/Tensor/LoggerTool_py/qq.txt','w',encoding='utf8');
count = 0;
contents = [];
tb = [16131007,561471062,672351023,46142027,63542006,24531013,349881004,502371010,5471001,976991020,190801006,621951002,108771061,197671005,40431061,1082461024,719661018,591821024,132221015,37171054,752431007,2571063,97491008,80031008,883721003,61541004,425991026,123261026,19941045,30662013,369881006,273901020,737451001,7301056,451821009,9691057,1176111019,235871008,681761012,1161451012,75311056,802311032,1075901016,277201001,29371005,826871025,604321061,312799001,24031001,433711043,786091024,255571014,46531005,56091005,1003651024,687431003,31841001,41481001,918591007,120531004,131451006,141001,22261020,918011003,158011009,214981001,795681026,605381029,11681019,194671052,1364441036,1206711031,39221002,634641020,467031063,1205731013,445481006,68841001,128831064,69051005,66301013,657201016,266941007,440261020,149631038,329221041,5622001,774891010,625781004,1082341027,928211015,816611004,1486751034,342091008,82381037,76401006,433801003,73571003,980971007,428291018
];
for line in fin.readlines():
    line = line.strip()
    finLines.append(line)
	
for id in tb:
    print(id%10000,round((id%10000)/1000))
    if round((id%10000)/1000) == 2:
        for line in finLines:
            line = line.replace('1027',str(int(id)%10000) )
            print(line,file=fout)
        print('\n',file=fout)
		
	    
fin.close()

list1 = []
# Python 2 默认以字节流（对应 Python 3 的 bytes）的方式读文件，不像 Python 3 默认解码为 unicode。
# 如果文件内容不是unicode编码的，要先以二进制方式打开，读入比特流，再解码。'rb'  #二进制方式打开


# 利用下面命令, 可以轻松吧制表符转换成为空格, MARK 一下. 
# sed -i 's/\t/  /g' *.py