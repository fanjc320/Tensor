

import numpy as np;
import codecs

#only first line
#a= np.loadtxt(fname = "./game.log",delimiter = '|')
#print(a)

# np.loadtxt(open("data.txt"), 'r',
           # dtype={
               # 'names': (
                   # 'sepal length', 'sepal width', 'petal length',
                   # 'petal width', 'label'),
               # 'formats': (
                   # np.float, np.float, np.float, np.float, np.str)}, #error
           # delimiter= ',', skiprows=0)

b = np.loadtxt("data.txt",
   dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'label'),
          'formats': (np.float, np.float, np.float, np.float, '|S15')},
   delimiter=',', skiprows=0)

print(b)

#各行item个数不同，这两个不能用，genfromtxt可以指定列名
#a= np.loadtxt(fname = "./game.log",encoding='utf8',delimiter = '|',dtype = np.str)
#c= np.genfromtxt(fname = "./game.log",encoding='utf8',delimiter = '|',dtype = np.str)
#print(c)

# with open(r'./game.log','r',encoding='utf8') as f:
    # d = f.read().splitlines()
	
#print("d:",d)

	
#print("d:",d)

# fo = open('./game.log','r',encoding='utf8')

# contents = [];
# for line in fo.readlines():
    # line = line.strip()
    # #print(line)
	
# fo.close()


fo = open('./game.log','r',encoding='utf8')

contents = [];
for line in fo.readlines():
    line = line.strip()
    print(line)
	
fo.close()

list1 = []
# 大文件读取
# https://blog.csdn.net/flyfrommath/article/details/73019961
with open('./game.log','r',encoding='utf8') as f:
    for line in f:
        oneline = line.split('|')
        print(oneline)
        list1.append(oneline)
		
#print('list1:',list1)

# h = os.system('tail -f ./game.log')
# print("h0:",h)
# title = h[0][0]
# print("title:",title)


# Python 2 默认以字节流（对应 Python 3 的 bytes）的方式读文件，不像 Python 3 默认解码为 unicode。
# 如果文件内容不是unicode编码的，要先以二进制方式打开，读入比特流，再解码。'rb'  #二进制方式打开


# 利用下面命令, 可以轻松吧制表符转换成为空格, MARK 一下. 
# sed -i 's/\t/  /g' *.py