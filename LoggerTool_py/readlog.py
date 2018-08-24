
import Xml_tlog
# import numpy as np;
# import codecs
import os
import time
import sys
# from colorama import Fore, Back, Style

from config import *
from colorama import init
# from termcolor import colored
init()

def prRed(skk,e='\n'): 
    print("\033[91m {}\033[00m" .format(skk),end=e)


def OutResult(content,split='',OutType = 1):
	if OutToFile == True:
		with open(OutLogPath,'a',encoding='utf8') as f:
			print(content,end=split,file=f)
	elif OutType == 1:
		print(content,end=split)
	elif OutType == 2:
		prRed(content,e=':')
		
	

with open(LogPath,'r',encoding='utf8') as f:
    f.seek(0,0)
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line:
            strs = line.split('|')
            tbname = strs[0] #表名
            if UseFilter and not tbname in TB_Filter:
                continue
            TB_List = []
            TB_List = Xml_tlog.main(tbname)
            for i in range(0,len(strs)):
                field = TB_List[i] #字段名
                if len(TB_List)>i:
                    if UseFilter and TB_Filter[tbname].count(field)==1:
                        # prRed(field,e=':')
                        OutResult(field,split=':',OutType=2)
                        
                    # macroname = Xml_tlog.macro(field)
                    # if macroname:
                        # print(macroname)
                    else:
                        # print(field,end=':')
                        OutResult(field,split=':',OutType=1)
                else:
                    # print("NULL",end=':')
                    OutResult("NULL",split=':',OutType=1)
                # print(strs[i],end=' ')
                OutResult(strs[i],split=' ',OutType=1)
            # print()
            OutResult("")
                    
        # time.sleep(0.5)#不可在cygwin下运行

		
# file.seek()方法标准格式是：seek(offset,whence=0)
# offset：开始的偏移量，也就是代表需要移动偏移的字节数
# whence：给offset参数一个定义，表示要从哪个位置开始偏移；0代表从文件开头开始算起，1代表从当前位置开始算起，2代表从文件末尾算起。
