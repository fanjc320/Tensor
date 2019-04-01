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
    sql = "SELECT * FROM pet9001"
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

# 二期补偿
dic_user_daoju = {}
cfgjson_daoju = {}
dic_cfg_daoju = {}
dic_user_daoju_detial = []

def HandleSqlRes(res):
    # print("res:",res)
    petjson = res["petjson"]
    text = json.loads(petjson)
    # print(text['pfd'])

    petid = int(res['petid'])
    uid = int(res['vRoleID'])
    petlv = int(res['petlv'])
    if(petid in [12,14,22,137]):
        # fout = open("./9001.txt", 'w', encoding='utf8');
        # print("userid:", res["vRoleID"], "petid:", res["petid"], "pfd:", text['pfd'])
        pfd = text['pfd']
        for k in pfd:
            prg = pfd[k]["prg"]#料理进度
            pos = pfd[k]["id"]#料理位置
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

                # 二期补偿
                exp = dic_cfg_daoju[petlv]

                daoju_numb_daoju = max(0,int((exp- 86760078)/1600000/1.75))+1
                print("exp daojuNumb::::",uid,petlv,exp,daoju_numb_daoju)
                if uid not in dic_user_daoju:
                    dic_user_daoju[uid] = daoju_numb_daoju
                else:
                    dic_user_daoju[uid] += daoju_numb_daoju

                dic_user_daoju_detial.push_back((uid,petlv,exp,daoju_numb_daoju))


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

    with open("./PetExpcfg.txt",encoding='utf-8') as f:
        cfgjson_daoju = json.loads(f.read())
        for k in cfgjson_daoju:
            # print("k::",k)
            vv = cfgjson_daoju[k]
            dic_cfg_daoju[vv['level']] = vv['exp']


    # print("dic-----",dic_cfg_daoju)
    conn = get_conn();
    readMysql(conn)
    # fout = open("./delpet_new.txt", 'w', encoding='utf8');
    # for k in dic_user:
    #     print(k[0],k[1],dic_user[k],file=fout)
    # fout.close()

# 二期补偿
    fout_daoju = open("./delpet_daoju.txt", 'w', encoding='utf8');
    for k in dic_user_daoju:
        print(k, dic_user_daoju[k], file=fout_daoju)
    fout_daoju.close()

    fout_daoju_detail = open("./delpet_daoju_detail.txt", 'w', encoding='utf8');
    for k in dic_user_daoju_detial:
        print(k, file=fout_daoju_detail)
    fout_daoju_detail.close()

if __name__ == '__main__':
    main()

     
