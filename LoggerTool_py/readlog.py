
import Xml_tlog
# import numpy as np;
# import codecs
import os
import time
import sys
# from colorama import Fore, Back, Style

from config import *
from colorama import init
from termcolor import colored
init()

def prRed(skk,e='\n'): 
    print("\033[91m {}\033[00m" .format(skk),end=e)

with open(LogPath,'r',encoding='utf8') as f:
    f.seek(0,2)
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line:
            strs = line.split('|')
            tbname = strs[0]
            if UseFilter and not tbname in TB_Filter:
                continue
            TB_List = []
            TB_List = Xml_tlog.main(tbname)
            for i in range(0,len(strs)):
                field = TB_List[i]
                if len(TB_List)>i:
                    if UseFilter and TB_Filter[tbname].count(field)==1:
                        prRed(field,e=':')
                    else:
                        print(field,end=':')
                else:
                    print("NULL",end=':')
                print(strs[i],end=' ')
            print()
                    
        time.sleep(0.5)#不可在cygwin下运行

