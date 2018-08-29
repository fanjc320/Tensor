import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
import wave

def TestClip1():
	a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
	b = a[:2, 1:3] #0-2行，1-3列 这是引用
	print("b",b)
	
	# A slice of an array is a view into the same data, so modifying it
	# will modify the original array.
	print(a[0,1])
	b[0,0]=77
	print(a[0,1])
	
def TestClip2():
	a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
	row_r1 = a[1,:]
	row_r2 = a[1:2,:]
	print("row_r1",row_r1,row_r1.shape,a.shape)#注意row_r1.shape(4,)
	print("row_r2",row_r2,row_r2.shape)
	
	col_r1 = a[:,1]
	col_r2 = a[:,1:2]
	print(col_r1,col_r1.shape)
	print(col_r2,col_r2.shape)

#索引为数值
TestClip3():
	a = np.array([[1,2],[3,4],[5,6]])
	print(a[[0,1,2],[0,1,0]]) # Prints "[1 4 5]"
	print np.array([a[0, 0], a[1, 1], a[2, 0]])  # Prints "[1 4 5]"
	# When using integer array indexing, you can reuse the same
	# element from the source array:
	print a[[0, 0], [1, 1]]  # Prints "[2 2]"  索引为（0,1） （0，1）
	# Equivalent to the previous integer array indexing example
	print np.array([a[0, 1], a[0, 1]])  # Prints "[2 2]"
	# Create a new array from which we will select elements
	a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
	
	print a  # prints "array([[ 1,  2,  3],
			#                [ 4,  5,  6],
			#                [ 7,  8,  9],
			#                [10, 11, 12]])"
	
	# Create an array of indices
	b = np.array([0, 2, 0, 1])
	
	# Select one element from each row of a using the indices in b
	print a[np.arange(4), b]  # Prints "[ 1  6  7 11]"
	
	# Mutate one element from each row of a using the indices in b
	a[np.arange(4), b] += 10
	
	print a  # prints "array([[11,  2,  3],
			#                [ 4,  5, 16],
			#                [17,  8,  9],
			#                [10, 21, 12]])
#索引为bool
def TestClip4():
	a = np.array([[1,2], [3, 4], [5, 6]])
	bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.
            
	print bool_idx      # Prints "[[False False]
						#          [ True  True]
						#          [ True  True]]"
	
	# We use boolean array indexing to construct a rank 1 array
	# consisting of the elements of a corresponding to the True values
	# of bool_idx
	print a[bool_idx]  # Prints "[3 4 5 6]"

	# We can do all of the above in a single concise statement:
	print a[a > 2]     # Prints "[3 4 5 6]"
			
def TestShow():
	time = np.arange(0,5,.05) # f=0.005 就是 1s 采样200个,共5s,就是1000个采样点
	x = np.sin(2*np.pi*1*time)
	# y = np.fft.fft(x)
	# show(x,y)
	# show(x,y,half=True)
	
	x2 = np.sin(2*np.pi*6*time)
	x3 = np.sin(2*np.pi*18*time)
	x += x2+x3
	y = np.fft.fft(x)
	show(x,y)

def Note():
	# 生成方波，振幅是 1，频率为 10Hz
	# 我们的间隔是 0.05s，每秒有 200 个点
	# 所以需要每隔 20 个点设为 1
	# x = np.zeros(len(time))
	# x[::20] = 1
	# print("x:",x);
	# y = np.fft.fft(x)
	# show(x, y)

	# 生成脉冲波
	# x = np.zeros(len(time)) 
	# x[380:400] = np.arange(0, 1, .05) 
	# x[400:420] = np.arange(1, 0, -.05) 
	# y = np.fft.fft(x) 
	# show(x, y) 
	
	# 生成随机数
	# x = np.random.random(100) 
	# y = np.fft.fft(x) 
	# show(x, y) 

	# TestShow()
	# range()函数
	
	# 函数说明： range(start, stop[, step]) -> range object，根据start与stop指定的范围以及step设定的步长，生成一个序列。
	# 参数含义：start:计数从start开始。默认是从0开始。例如range（5）等价于range（0， 5）;
				# end:技术到end结束，但不包括end.例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
				# scan：每次跳跃的间距，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
	# 函数返回的是一个range object
	# >>> range(0,5)                 #生成一个range object,而不是[0,1,2,3,4]   
	# range(0, 5)     
	# >>> c = [i for i in range(0,5)]     #从0 开始到4，不包括5，默认的间隔为1  
	# >>> c  
	# [0, 1, 2, 3, 4]  
	
	# arrange()函数
	
	# 函数说明：arange([start,] stop[, step,], dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。 dtype : dtype
	# >>> np.arange(3)  
	# array([0, 1, 2])  
	# >>> np.arange(3.0)  
	# array([ 0.,  1.,  2.])  
	# >>> np.arange(3,7)  
	# array([3, 4, 5, 6])  
	# >>> np.arange(3,7,2)  
	# array([3, 5])
	print("")

TestClip3()