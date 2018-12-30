import numpy as np

def Dot():
    # dot()返回的是两个数组的点积(dot product)
    # 1.如果处理的是一维数组，则得到的是两数组的內积（顺便去补一下数学知识）
    d = np.arange(0,9)
    print(d)
    e = d[::-1]
    print(e)
    print(np.dot(d,e))
    # 如果是二维数组（矩阵）之间的运算，则得到的是矩阵积（mastrix product）。

    a = np.arange(1,5).reshape(2,2)
    print(a)
    b = np.arange(5,9).reshape(2,2)
    print(b)
    print(np.dot(a,b))

Dot();
# 3.dot()函数可以通过numpy库调用，也可以由数组实例对象进行调用。a.dot(b) 与 np.dot(a,b)效果相同。
# 矩阵积计算不遵循交换律,np.dot(a,b) 和 np.dot(b,a) 得到的结果是不一样的。



class Student():
    @property
    def myscore(self):
        return self._score,self.test

    @myscore.setter
    def myscore(self,value):
        if not isinstance(value,int):
            raise ValueError('score must be an integer!')
        if value <0 or value >100:
            raise ValueError('score must between 0~100!')
        self._score = value;
        self.test = value+1;

s = Student()
s.myscore = 60
# print(s.score)
print(s.myscore)
