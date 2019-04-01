#-*- coding: UTF-8 -*-
# import copy
import pymysql
import json


def get_conn():
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='宝石_0329', charset='utf8')
    # conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='test', charset='utf8')
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
    sql = "SELECT * FROM finaldata"
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

first = {}
second = {}
third = {}
dic_user = {}
dic_result ={}

# 统一configid
def HandleSqlRes(res):

    time = res["dteventtime"]
    openid = res['vopenid']
    svrid= int(res['gamesvrid'])
    vroleid = int(res['vroleid'])
    itemid = int(res['igoodsidreal'])
    addorreduce = int(res['addorreduce'])
    aftercount = int(res['aftercount'])
    reason = int(res['reason'])
   
    item = (addorreduce,itemid,aftercount,reason,vroleid,time,svrid)

    global first,second,third,dic_user,dic_result
    if first==None:
        first ={}
    if second==None:
        second={}
    if third==None:
        third={}
    if dic_user==None:
        dic_user={}
    if dic_result==None:
        dic_result={}

    if addorreduce == 1:# 第一条碰到的是减少
        if first.get(vroleid)==None:
            first[vroleid]=item
            # print("00   aa")
        elif first.get(vroleid)!=None:
            first[vroleid]=item
            # print("00   bb")
    else:
        if second.get(vroleid)==None:#没有第二条
            if first.get(vroleid)==None:
                # print("--00--")
                pass
            elif first.get(vroleid)!=None:
                # print("--- first:",first)
                first_aftercount = first[vroleid][2]
                first_itemid = first[vroleid][1]
                if aftercount-first_aftercount==1 and itemid==first_itemid: #满足第二条的表现是比上一条多1，且道具id相同
                    second[vroleid]=item
                    # print("11   aa ",first)
                else:
                    first[vroleid]=None
                    # print("11   bb")
            return

        
        if second.get(vroleid)!=None:#有第二条，才看有没有第三条
            second_aftercount = first[vroleid][2]
            second_time =first[vroleid][5]
            second_itemid=first[vroleid][1]
            # print("---------second:",aftercount-second_aftercount,time==second_time,itemid==second_itemid)
            if aftercount-second_aftercount==1 and time==second_time and itemid==second_itemid:#满足第三条的条件是比第二条多1，道具id相同，时间相同
               third[vroleid]=item
               # print("--------------22   aa")
               first[vroleid] = None
               second[vroleid] = None
        else:
            first[vroleid]=None
            second[vroleid]=None
            # print("22   bb")

    if third.get(vroleid)!=None:
        # print(first,second,third)
        dic_user[vroleid]=(first,second,third)
        dic_result[vroleid]=(svrid,openid,vroleid)
        first=None
        second=None
        third=None


def main():
    # print("dic-----",dic_cfg_daoju)
    conn = get_conn();
    readMysql(conn)
    
    for k in dic_result:
        print(dic_result[k][0],"\t",dic_result[k][1],"\t",dic_result[k][2])

    fout_daoju = open("./dic_user.txt", 'w', encoding='utf8');

    # for k in dic_user:
    #     print(k, dic_user[k], file=fout_daoju)
    # fout_daoju.close()


if __name__ == '__main__':
    main()

     
