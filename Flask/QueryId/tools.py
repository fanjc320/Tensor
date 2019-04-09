# coding=utf8
import pymysql

def my_db(sql):
    conn=pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='123456',
        db='宝石_0329',
        charset='utf8',
        autocommit=True
    )

    cur=conn.cursor(cursor=pymysql.cursors.DictCursor)# 建立游标；默认返回二维数组，DictCursor指定返回字典；
    print("sql:",sql)
    cur.execute(sql)
    res = cur.fetchall()
    cur.close()
    conn.close()
    return res