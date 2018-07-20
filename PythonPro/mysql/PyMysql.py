import pymysql.cursors

# 插入
# 连接MySQL数据库
connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='tlog1', 
                             charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

# 通过cursor创建游标
cursor = connection.cursor()

# 创建sql 语句，并执行
sql = "INSERT INTO `users` (`email`, `password`) VALUES ('huzhiheng@itest.info', '123456')"
cursor.execute(sql)

# 提交SQL
connection.commit()


# 查询
# 连接MySQL数据库
connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='198876', db='guest', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)


# 通过cursor创建游标
cursor = connection.cursor()

# 执行数据查询
sql = "SELECT `id`, `password` FROM `users` WHERE `email`='huzhiheng@itest.info'"
cursor.execute(sql)

#查询数据库单条数据
result = cursor.fetchone()
print(result)

print("-----------华丽分割线------------")

# 执行数据查询
sql = "SELECT `id`, `password` FROM `users`"
cursor.execute(sql)

#查询数据库多条数据
result = cursor.fetchall()
for data in result:
    print(data)


# 关闭数据连接
connection.close()