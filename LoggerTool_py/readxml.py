
import pymysql
import csv
import codecs
import Xml_tlog
import datetime

G_TB_List = []

def createTb(table_name,withhead=1):
    head = []
    if withhead:
        head = ['tdbank_imp_date','worldid','ip']
    print("===G_TB_List.size:",len(G_TB_List),"G_tb_list:",G_TB_List)
    head.extend(G_TB_List)
    print("===newlist.size:",len(head),"newlist:",head)



if __name__ == '__main__':
    tablename = "ItemFlow"
    G_TB_List = Xml_tlog.main(tablename)
    createTb(tablename)	