# coding=utf-8

# import  tensorflow as tf
# initial = tf.truncated_normal(shape=[10,10],mean=0,stddev=1)
# W= tf.Variable(initial)
# list=[[1.,1.],[2.,2.]]
# X = tf.Variable(list,dtype=tf.float32)
# ini_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(ini_op)
#     print("run x:",sess.run(W[:2,:2]))
#
#     op=W[:2,:2].assign(22.*tf.ones((2,2)))
#     print("run z:",sess.run(op))
#     print("w.eval:",W.eval())
#     print("##############6###########")
#     print("w.dtype:",W.dtype)
#     print("run a:",sess.run(W.initial_value))
#     print("run b:",sess.run(W.op))
#     print("shape:",W.shape)
#     print("##############7############")
#     print("run c:",sess.run(X))


import tensorflow as tf

# Model parameters（原始参数）
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)


# Model input and output（占位符，声明变量）
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss  （损失函数，相减的平方再求和）
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer  （最小下降法的步长为0.01）
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data    （数据）
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop  （训练过程）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))