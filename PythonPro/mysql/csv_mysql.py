#mysql 3.9 seonds
#LOAD DATA LOW_PRIORITY LOCAL INFILE 'E:\\TestProjs\\TestPython\\roundflow.csv' REPLACE INTO TABLE `t_csv` CHARACTER SET latin1 FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';
#crate mysql table from tlog_fields.xml and  turn csv into mysql  csv(58M,257294 lines) use 47senconds
#用 load data 60万行导入需要80s 
import pymysql
import csv
import codecs
import Xml_tlog
import datetime

G_TB_List = []

def get_conn():
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='test_csv', charset='utf8')
    return conn

def createTb(table_name,withhead=1):
    head = []
    if withhead:
        head = ['tdbank_imp_date','worldid','ip']
    print("===G_TB_List.size:",len(G_TB_List),"G_tb_list:",G_TB_List)
    head.extend(G_TB_List)
    print("===newlist.size:",len(head),"newlist:",head)
    createsqltable = """CREATE TABLE IF NOT EXISTS """ + table_name + " (" + " VARCHAR(250),".join(head) + " VARCHAR(250))"
    #print createsqltable
    conn = get_conn()
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
    tablename = "ItemFlow"
    G_TB_List = Xml_tlog.main(tablename)
    createTb(tablename)
    #read_csv_to_mysql('E:\\TestProjs\\TestPython\\roundflow.csv') #47 seconds
    #read_csv_to_mysql('E:\\TestProjs\\TestPython\\test.csv')	