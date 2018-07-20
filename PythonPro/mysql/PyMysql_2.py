#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql

config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'passwd': '123456',
    'charset':'utf8mb4',
    'cursorclass':pymysql.cursors.DictCursor
    }
conn = pymysql.connect(**config)
conn.autocommit(1)
cursor = conn.cursor()

try:
    # �������ݿ�
    DB_NAME = 'test'
    cursor.execute('DROP DATABASE IF EXISTS %s' %DB_NAME)
    cursor.execute('CREATE DATABASE IF NOT EXISTS %s' %DB_NAME)
    conn.select_db(DB_NAME)

    #������
    TABLE_NAME = 'user'
    cursor.execute('CREATE TABLE %s(id int primary key,name varchar(30))' %TABLE_NAME)

    # ���������¼
    values = []
    for i in range(20):
        values.append((i,'kk'+str(i)))
    cursor.executemany('INSERT INTO user values(%s,%s)',values)

    # ��ѯ������Ŀ
    count = cursor.execute('SELECT * FROM %s' %TABLE_NAME)
    print ('total records:', cursor.rowcount)

    # ��ȡ������Ϣ
    desc = cursor.description
    print ("%s %3s" % (desc[0][0], desc[1][0]))

    cursor.scroll(10,mode='absolute')
    results = cursor.fetchall()
    for result in results:
        print (result)

except:
    import traceback
    traceback.print_exc()
    # ��������ʱ���
    conn.rollback()
finally:
    # �ر��α�����
    cursor.close()
    # �ر����ݿ�����
    conn.close()