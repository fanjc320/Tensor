# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''双层神经网络实现mnist'''

mnist = input_data.read_data_sets("MNIST_data/",one_hot= True)

dim_input = 784
neural_num_hidden_1 = 256
neural_num_hidden_2 = 128
dim_input = 10


x_data = tf.placeholder(tf.float32,[None,dim_input]) #样本个数不确定，所以None
y_real = tf.placeholder(tf.float32,[None,dim_input])

stddev = 0.1
weights = {"w1":tf.Variable(tf.random_normal([dim_input,neural_num_hidden_1],stddev=stddev)),
           "w2":tf.Variable(tf.random_normal([neural_num_hidden_1,dim_input],stddev=stddev)),
           "w_out":tf.Variable(tf.random_normal([neural_num_hidden_2,dim_output],stddev=stddev))}
biases = {"b1":tf.Variable(tf.zeros([neural_num_hidden_1])),
          "b2":tf.Variable(tf.zeros(neural_num_hidden_2)),
          "b_out":tf.Variable(tf.zeros([dim_output]))}

print("网络参数：",weights,biases)


def forward_prop(X_input,_weights,_biases):
    """
        通过网络前向传播预测
        :param X_input: 输入
        :param _weights: 权重
        :param _biases: 偏置
        """
    layer_1 = tf.nn.relu(tf.add(tf.matmul(X_input,_weights["w1"]),_biases["b1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,_weights["w2"]),_biases["b2"]))
    return tf.add(tf.matmul(layer_2,_weights["w_out"]),_biases["b_out"])

y_pred = forward_prop(x_data,weights,biases) #预测值

# 定义loss函数以及优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_real))
op= tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

# 定义准确率
correct = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_real, 1))  # 计算出预测值与真实值是否相等(独热码为1的索引是否相等)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 每个batch的平均准确率 将True转成float

# 训练参数
training_epoch = 100
batch_size = 128
display_step = 2  # 设置打印间隔

# =========《准备训练测试》==========
init = tf.global_variables_initializer()

total_batch = mnist.train.num_examples//batch_size #" / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。
print("共有%d个batch,每个batch大小为:%d"%(total_batch,batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoch):
        avg_loss = 0 #存储所有batch平均loss值

        for i_batch in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {x_data:batch_xs,y_real:batch_ys}
            sess.run(op,feed_dict=feed_dict)
            avg_loss+=sess.run(loss,feed_dict=feed_dict)

        avg_loss = avg_loss/total_batch


        if epoch % display_step == 0:
            print("Epoch:%3d/%3d,loss:%.6f"%(epoch,training_epoch,avg_loss))

            feed_dict = {x_data:batch_xs,y_real:batch_ys}
            train_accuracy = sess.run(accuracy,feed_dict=feed_dict)
            print("训练准确率:%.6f"%train_accuracy)

            feed_dict = {x_data:mnist.test.images,y_real:mnist.test.labels}
            test_accuracy = sess.run(accuracy,feed_dict=feed_dict)
            print("测试准确率:%.6f"%test_accuracy)

print("over!")
