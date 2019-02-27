#-*- coding: UTF-8 -*-
# import copy
import pymysql
import json


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

def readMysql(conn):
    # cursor = conn.cursor()
    # 默认情况下，我们获取到的返回值是元组，只能看到每行的数据，却不知道每一列代表的是什么，这个时候可以使用以下方式来返回字典，每一行的数据都会生成一个字典：
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 在实例化的时候，将属性cursor设置为pymysql.cursors.DictCursor
    # sql = "SELECT vRoleID,petid,petjson FROM delpet9001"
    sql = "SELECT * FROM delpet"
    cursor.execute(sql)
    # res = cursor.fetchall()
    while(True):
        res = cursor.fetchone()  # 第一次执行
        if(res==None):
            break
        # print(res)
        HandleSqlRes(res)

    print("============END===============")
    cursor.close()
    conn.close()


dic_cfg = {

}
dic_user = {

}
cfgjson = {}
def HandleSqlRes(res):
    petjson = res["petjson"]
    text = json.loads(petjson)
    # print(text['pfd'])

    petid = int(res['petid'])
    uid = int(res['vRoleID'])
    if(petid in [12,14,22,137]):
        # fout = open("./9001.txt", 'w', encoding='utf8');
        # print("userid:", res["vRoleID"], "petid:", res["petid"], "pfd:", text['pfd'])
        pfd = text['pfd']
        for k in pfd:
            prg = pfd[k]["prg"]
            pos = pfd[k]["id"]
            if prg !=0:
                item =dic_cfg.get((petid,pos))
                daoju_id = item[0]
                daoju_numb = item[1]
                # print("====",daoju_id,daoju_numb)
                # dic_user.setdefault(uid,(daoju_id))
                if (uid,daoju_id) not in dic_user:
                    dic_user[(uid,daoju_id)] = daoju_numb
                else:
                    dic_user[(uid, daoju_id)] += daoju_numb
                # print("prg:", prg)
                # print("test:",cfgjson[])
                # print("uid:", res["vRoleID"], "petid:", prg, file=fout)


        # print("userid:",res["vRoleID"],"petid:",res["petid"],"pfd:",text['pfd'],file=fout)
        # fout.close()


def main():
    # cfgjson = json.load("./Petfoodcfg.txt")
    with open("./Petfoodcfg.txt",encoding='utf-8') as f:
        cfgjson = json.loads(f.read())
        for k in cfgjson:
            for it in cfgjson[k]:
                # print("item:", it)
                dic_cfg[(it['id'],it['position'])] = (it['food'],it['foodnumber'])

        # print("dic_cfg:",dic_cfg)
    conn = get_conn();
    readMysql(conn)
    fout = open("./delpet.txt", 'w', encoding='utf8');
    for k in dic_user:
        print(k[0],k[1],dic_user[k],file=fout)
    fout.close()

if __name__ == '__main__':
    main()

     
