#-*- coding: UTF-8 -*-

import xml.sax
import copy
from config import XmlPath
import pymysql

G_list =[];

class NodeHandler(xml.sax.ContentHandler):
	def __init__(self):
		index = 1
		self.key = "Begin"
		self.all = {}
		self.macros = {}
		self.content = []
		
	def printall(self):
			with open("xmltables.txt".format(),'w') as f:
				for k,v in self.all.items():
					newlist = "	".join(v)
					print(newlist,file=f)
		

	def startElement(self,tag,attributes):
		#print("start_tag----------:",tag);
		key = attributes["name"]
		if(tag=="struct"):
			self.key = key
		if(tag=="macrosgroup"):
			self.key = key
		
		self.content.append(key)
		
	def endElement(self,tag):
		#print("-------------end_tag ",tag)
		if(tag=="struct"):
			self.all[self.key] = copy.deepcopy(self.content)
			self.content.clear()
		if(tag=="macrosgroup"):
			self.macros[self.key] = copy.deepcopy(self.content)
			self.content.clear()
		
# 内容事件处理
	#def characters(self, content):
		#print("self.key:",self.key)

Handler = NodeHandler()
def main(tablename):
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces,0)
	# Handler = NodeHandler()
	parser.setContentHandler( Handler )
	parser.parse(XmlPath)
	
	Handler.printall()
	
	G_list =copy.deepcopy( Handler.all.get(tablename) )
	return G_list
		
def macro(macroname):
	return copy.deepcopy( Handler.macros.get(macroname) )
		
		
		
		
#mysql 3.9 seonds
#LOAD DATA LOW_PRIORITY LOCAL INFILE 'E:\\TestProjs\\TestPython\\roundflow.csv' REPLACE INTO TABLE `t_csv` CHARACTER SET latin1 FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';
#crate mysql table from tlog_fields.xml and  turn csv into mysql  csv(58M,257294 lines) use 47senconds
#用 load data 60万行导入需要80s 
import pymysql
import csv
import codecs
import Xml_tlog
import datetime

# G_TB_List = []

def get_conn():
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='tlog_t', charset='utf8')
    return conn

def createTb(conn,G_TB_List,withhead=0,drop = 1):
    for k,v in G_TB_List.items():
        newlist = "	".join(v)
        # print(k,v)
        head = []
        if withhead:
            head = ['tdbank_imp_date','worldid','ip']
        # print("===G_TB_List.size:",len(G_TB_List),"G_tb_list:",G_TB_List)
        head.extend(v)
        if drop:
            createsqltable = "DROP TABLE IF EXISTS "+k+";";
            cur = conn.cursor()
            cur.execute(createsqltable)
            conn.commit()
        createsqltable = "CREATE TABLE IF NOT EXISTS " + k + " (" + " VARCHAR(250),".join(head) + " VARCHAR(250))"
        # createsqltable += "ENGINE=InnoDB DEFAULT CHARSET=utf8"
        print(createsqltable)
        
        cur = conn.cursor()
        cur.execute(createsqltable)
        conn.commit()

def insert(cur, sql, args):
    cur.execute(sql, args)
	
def read_csv_to_mysql(filename):
    begintime = datetime.datetime.now()
    print("begintime:",begintime)
    with codecs.open(filename=filename, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        head = next(reader)
        conn = get_conn()
        cur = conn.cursor()
        print("head:",head,"cur:",cur)
        sql = 'insert into t_csv values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s)'
        for item in reader:
            #print("item[1]:",item)
            if item[1] is None or item[1] == '':  # item[1]作为唯一键，不能为null
                continue
            args = tuple(item)
            #print("args:",args)
            #print("curl:",cur)
            insert(cur, sql=sql, args=args)

        conn.commit()
        cur.close()
        conn.close()
    endtime = datetime.datetime.now()
    print("endtime:",endtime)
    print("timespan:",endtime-begintime)

if __name__ == '__main__':
    # tablename = "ItemFlow"
    # G_TB_List = Xml_tlog.main("")
    G_TB_List = Xml_tlog.getAll()
    conn = get_conn();
    createTb(conn,G_TB_List)
    # createTb(tablename)
    #read_csv_to_mysql('E:\\TestProjs\\TestPython\\roundflow.csv') #47 seconds
    #read_csv_to_mysql('E:\\TestProjs\\TestPython\\test.csv')	
		
if ( __name__ == "__main__" ):
	main("")