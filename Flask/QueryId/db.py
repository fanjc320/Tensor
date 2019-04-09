# encoding: utf-8
from flask import Flask
import sqlite3


def get_data(id):
    conn = sqlite3.connect('./db/course.db')
    # 查询语句

    id = int(id)
    query_sql = '''
SELECT
*
FROM
table_name  #修改
WHERE
CUST_CODE = %d
'''

    ##进行查询
    tem = []

    query = conn.execute(query_sql % id)
    for i in query:
        tem.append(i)

    conn.close()
    return  tem