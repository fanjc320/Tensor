import pymysql.cursors

# ����
# ����MySQL���ݿ�
connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='tlog1', 
                             charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

# ͨ��cursor�����α�
cursor = connection.cursor()

# ����sql ��䣬��ִ��
sql = "INSERT INTO `users` (`email`, `password`) VALUES ('huzhiheng@itest.info', '123456')"
cursor.execute(sql)

# �ύSQL
connection.commit()


# ��ѯ
# ����MySQL���ݿ�
connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='198876', db='guest', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)


# ͨ��cursor�����α�
cursor = connection.cursor()

# ִ�����ݲ�ѯ
sql = "SELECT `id`, `password` FROM `users` WHERE `email`='huzhiheng@itest.info'"
cursor.execute(sql)

#��ѯ���ݿⵥ������
result = cursor.fetchone()
print(result)

print("-----------�����ָ���------------")

# ִ�����ݲ�ѯ
sql = "SELECT `id`, `password` FROM `users`"
cursor.execute(sql)

#��ѯ���ݿ��������
result = cursor.fetchall()
for data in result:
    print(data)


# �ر���������
connection.close()