# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:52:56 2018

@author: lenovo
"""

import numpy as np
import math
import tensorflow as tf
import struct
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import os

def LoadWave(filepath,offset,buf,buflen):
    f = open(filepath, 'rb')  
    ndx = 0
    f.seek(offset*2)
    while True:  
        temp=f.read(2)  
        if len(temp) == 2:  
            value = struct.unpack('h',temp)
            buf[ndx] = int(value[0])
            ndx += 1
            if (ndx == buflen):break
        else:  
            break  
    f.close()

def SaveWave(filepath,buf,repeat,append):
    if append:opt = 'ab' 
    else:opt = 'wb'

    f = open(filepath, opt)
    for n in range(repeat):
        buf= buf.astype(np.int16)
        f.write(buf)   
    f.close()

def SaveData(filepath,buf):
    f = open(filepath, 'wb')
    buf= buf.astype(np.float32)
    f.write(buf)   
    f.close()    
    
def LoadData(filepath,buf):    
    f = open(filepath, 'rb')  
    ndx = 0
    while True:  
        temp=f.read(4)  
        if len(temp) == 4:  
            value = struct.unpack('f',temp)
            buf[ndx] = float(value[0])
            ndx += 1
        else:  
            break    
    f.close()
    
def show_one(wave,color):
    plt.title("wave", fontsize=14)
    instances = range(len(wave))
    plt.plot(instances, wave[instances], marker='.', markersize=1, c=color)
    plt.xlabel("test")   
    
#返回：每一行是x_delay个x对应1个y，总共batch_size行
def sample_get(wave_y,y_delay):
    ret_y = np.reshape(wave_y,[1,len(wave_y)])
    ret_yl = np.zeros([1,y_delay],dtype='float32')
    return ret_y,ret_yl

GlottisTime = 74
N = 30
M = 30  
batch_size = 1

def cond(n, x, yl,y,Wa,Wb):
    return n < GlottisTime

def body(n, x, yl,y,Wa,Wb):

    #x切片    
    x_delay = tf.reshape(tf.slice(x,[0,n],[batch_size,M]),[batch_size,M]) 
    
    #yl_切片
    y_delay = tf.reshape(yl,[batch_size,N])
        
    #递归方程计算出当前y值
    y_cur = tf.subtract(tf.matmul(x_delay,Wb),tf.matmul(y_delay,Wa)) 
    y_cur = tf.reshape(y_cur,[batch_size,1])
        
    #当前y值填充到输出序列中的第n个位置
    y_out = tf.concat([tf.slice(y,[0,0],[batch_size,n]),tf.to_float(y_cur)],axis=1) 
    y_out = tf.concat([y_out,tf.slice(y,[0,n+1],[batch_size,GlottisTime-n-1])],axis=1) 
    y_out.set_shape([batch_size,GlottisTime])
    
    #yl的0..N-1位向后移1位,当前输出值保存到yl的第0位
    yl = tf.concat([y_cur,tf.slice(yl,[0,0],[batch_size,N-1])],axis=1)    
    
    return n+1, x, yl,y_out,Wa,Wb
        
def main(param_mode,param_type,param_voice):
            
    #原始波形
    wave_x_base = np.zeros([GlottisTime],dtype='int16')   
    wave_y_base = np.zeros([GlottisTime],dtype='int16')  
    
    #加载x语音
    LoadWave('../data/std/m74.snd',0,wave_x_base,GlottisTime)
    
    #加载y语音
    path = '../data/std/{}74.snd'.format(param_voice)
    LoadWave(path,0,wave_y_base,GlottisTime)          
        
    #显示x,y波形
    show_one(wave_x_base,'r')    
    show_one(wave_y_base,'b')
        
    #return
    
    tf.reset_default_graph()
    ##### 输入层 #####

    Yd = tf.placeholder(tf.float32, [None,GlottisTime])       #目标输出
    Yl = tf.placeholder(tf.float64, [None,N])                 #Y历史输出
    Ki = tf.placeholder(tf.float32, [GlottisTime],name='Ki')  #FIR系数输入
    Wai = tf.placeholder(tf.float32, [N],name='Wai')  #IIR-a系数输入    
    Wbi = tf.placeholder(tf.float32, [M],name='Wbi')  #IIR-b系数输入    

    ###### FIR层 输入必须是32位#####
    Kv = tf.Variable(tf.zeros([GlottisTime]), dtype=tf.float32) #卷积核变量    
    
    if param_mode=='Train' and param_type=='FIR' :
        K = Kv         
    else:
        K = tf.cast(Ki,dtype=tf.float32)  

    if param_mode=='Train':init_K_Ki = tf.assign(K,Ki)
                            
    #因果滤波器0对称补零                        
    K_ = tf.concat([K,tf.zeros([GlottisTime-1])],axis=0)
    K_ = tf.reshape(K_,[2*GlottisTime-1,1,1]) 
    #构造脉冲 [ 1,GlottisTime-1个0 ] 
    P = tf.concat([tf.ones([1]),tf.zeros([GlottisTime-1])],axis=0)    
    P = tf.reshape(P,[1,GlottisTime,1])
    #卷积输出（模拟声门波）
    X = tf.nn.conv1d(P, K_, 1, 'SAME') 
    X = tf.reshape(X,[1,GlottisTime])
    X = tf.concat([tf.zeros([1,M-1]),X],axis=1)    #（前插M-1个0）    
    X = tf.cast(X,dtype=tf.float64) 
        
    ###### IIR层 输入必须是64位#####
    Wbv = tf.cast(tf.Variable(tf.zeros([M,1]),dtype=tf.float32),dtype=tf.float64) 
    Wav = tf.cast(tf.Variable(tf.zeros([N,1]),dtype=tf.float32),dtype=tf.float64) 
    
    if param_mode=='Train' and param_type=='IIR' :
        Wb = Wbv
        Wa = Wav    
    else:
        Wb = tf.cast(tf.reshape(Wbi,[M,1]),dtype=tf.float64)  
        Wa = tf.cast(tf.reshape(Wai,[N,1]),dtype=tf.float64) 
        
    n_loop = tf.Variable(0, dtype=tf.int32)                         #循环计数
    Yout = tf.Variable(tf.zeros([batch_size,GlottisTime]),dtype=tf.float32)   #IIR输出
    n_loop_, X_, Yl_,Yout_,Wa_,Wb_ = tf.while_loop(cond, body, [n_loop, X, Yl,Yout,Wa,Wb]) #循环图
       
    ####### 训练方法 #################
    loss = tf.reduce_mean(tf.square(Yout_ - tf.to_float(Yd)))
    FLAGS_model_ckpt_path = './model/{}_{}/mymodel.ckpt'.format(param_type,param_voice)   
    learning_rate = 1.1
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
        
    ####### sesson ###############
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    iteration = 0

    with tf.Session() as sess:
        
        if os.path.exists(FLAGS_model_ckpt_path+'.meta'):
            #载入预训练模型        
            ckpt_path = FLAGS_model_ckpt_path
            saver = tf.train.import_meta_graph(ckpt_path + '.meta')
            saver.restore(sess, ckpt_path)
        else:
            sess.run(init) #全新训练的初始化       
        #sess.run(init) #全新训练的初始化       

        #取样
        Y_batch,Yl_batch = sample_get(wave_y_base,N) 
        
        #从文件导入Wa,Wb
        wa_data = np.zeros([N],dtype='float32')
        wb_data = np.zeros([M],dtype='float32') 
        LoadData('../data/wa.dat',wa_data)
        LoadData('../data/wb.dat',wb_data) 
        #从文件导入k
        glot_wav = np.zeros([GlottisTime],dtype='int16') 
        LoadWave('../data/m74.snd',0,glot_wav,GlottisTime)  
        k_data = glot_wav[::-1].astype(np.float32)
        
        feed = {Yd:Y_batch,Yl:Yl_batch,Ki:k_data,Wai:wa_data,Wbi:wb_data}

        #todo 训练之前先初始化   
        if param_mode=='Train':sess.run(init_K_Ki,feed_dict=feed)
        
        while(param_mode == 'Train'):
            iteration += 1
            
            sess.run(training_op, feed_dict=feed)
            
            wa = sess.run(Wa, feed_dict=feed)
            wb = sess.run(Wb, feed_dict=feed)
            k = sess.run(K, feed_dict=feed)
                    
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict=feed)
                print(iteration, "\tMSE:", mse)
            
                #print('k',k)
                #print('a',wa)
                #print('b',wb)            
            
            
            if iteration % 100 == 0:
                #保存Wa,Wb到文件
                SaveData('../data/wa.dat',wa)
                SaveData('../data/wb.dat',wb)
                #保存k到文件
                glot_wav = k[::-1].astype(np.int16)
                SaveWave('../data/k.snd',glot_wav,1,False)
                
                saver.save(sess, FLAGS_model_ckpt_path)
                print("\tmodel saved.")
                iteration = 0

        ######## 预测 ####################
        Yl_batch = np.zeros([1,N]) 
        res = sess.run([n_loop_, X_, Yl_,Yout_,Wa_,Wb_],feed_dict=feed)
        wave_pred =  res[3][0]                   
        show_one(wave_pred,'y') #预测显示
        
        Xv = sess.run(X, feed_dict=feed)
        Pv = sess.run(P, feed_dict=feed)
        K_v = sess.run(K_, feed_dict=feed)
        #show_one(Xv,'g') #预测显示
        
        # 增益递减 fooplot.com 1-x/1.1^(60-x)
        rate = np.zeros([30],dtype='float32')
        for n in range(30):
            rate[n] = 1-n/(1.1**(60-n))
            if (rate[n] < 0):rate[n] = 0
        
        # 连续递减保存 
        bAppend = False
        for n in range(30):                        
            wave_pred =  res[3][0] *rate[n]                                
            SaveWave('../data/gen_{}.snd'.format(param_voice),wave_pred,5,bAppend)
            bAppend = True
        

    
    


if __name__ == '__main__':
    mode = 'Train'
    #mode = 'Test'

    #type = 'IIR'
    type = 'FIR'

    voice = 'a'
    
    main(mode,type,voice)      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    