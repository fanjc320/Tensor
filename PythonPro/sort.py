
#sorted(iterable[, cmp[, key[, reverse]]])

a = [5,7,6,3,4,1,2]
b = sorted(a)
print(a)
print(b)

L = [('b', 2), ('a', 1), ('c', 3), ('d', 4)]
sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))# 利用cmp函数
print(L)
sorted(L, key=lambda x:x[1]) # 利用key
print(L)

students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
s1 = sorted(students, key=lambda s: s[2])            # 按年龄排序
print(s1)
s2 = sorted(students, key=lambda s: s[2], reverse=True)       # 按降序
print(s2)

s3 = sorted(students, key=lambda student : student[2])   # sort by age
print(s3)
s4 = sorted(students, cmp=lambda x,y : cmp(x[2], y[2])) # sort by age
print(s4)


#用 operator 函数来加快速度,
from operator import itemgetter, attrgetter
s5 = sorted(students, key=itemgetter(2))

#用 operator 函数进行多级排序
s6 = sorted(students, key=itemgetter(1,2))  # sort by grade then by age
print(s6)

#2. 对由字典排序 ，返回由tuple组成的List,不再是字典。
d = {'data1':3, 'data2':1, 'data3':2, 'data4':4}
d1 = sorted(d.iteritems(), key=itemgetter(1), reverse=True)
print(d1)

