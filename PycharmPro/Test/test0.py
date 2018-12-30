import tensorflow as tf
import numpy as np

def TestVariable():
    A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(A))
        sess.run(tf.assign(A, 10))
        print(sess.run(A))

def TestRef(v1,v2,v3):
    v1= 10
    v2[1]=15
    v3[0]=30

def TestSlice():
    x = [[1, 2, 3], [4, 5, 6],[7,8,9]]
    y = np.arange(24).reshape([2, 3, 4])
    # z = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
    z = tf.constant([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]]
    ])
    # sess = tf.Session()
    with tf.Session() as sess:
        begin_x = [1, 0]  # 第一个1，决定了从x的第二行[4,5,6]开始，第二个0，决定了从[4,5,6] 中的4开始抽取
        size_x = [1, 2]  # 第一个1决定了，从第二行以起始位置抽取1行，也就是只抽取[4,5,6] 这一行，在这一行中从4开始抽取2个元素
        out = tf.slice(x, begin_x, size_x)
        print(sess.run(out))  # 结果:[[4 5]]

        begin_y = [1, 0, 0]
        print("z100",sess.run(z[1][0][0]))
        # size_y = [1, 2, 3]
        size_y = [1, 1, 1]
        out = tf.slice(y, begin_y, size_y)
        print(sess.run(out))  # 结果:[[[12 13 14] [16 17 18]]]

        print("----------------")
        begin_z = [0, 1, 1]
        size_z = [-1, 1, 2]
        out = tf.slice(z, begin_z, size_z)
        print(sess.run(out))  # size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取，结果：[[[ 5  6]] [[11 12]] [[17 18]]]

# TestSlice()
#
#
# print("test ref and value as param")
# value1 = 1;value2 = np.zeros([10],int);value3 = [3,3,3,3,3]
# TestRef(value1,value2,value3);
# print("value:",value1,value2,value3)#穿参时,列表是作为引用,普通数值是复制


class FizzBuzz():
    def __init__(self, length=30):
        self.length = length  # 程序需要执行的序列长度
        self.array = tf.Variable([str(i) for i in range(1, length + 1)], dtype=tf.string, trainable=False)  # 最后程序返回的结果
        self.graph = tf.while_loop(self.cond, self.body, [1, self.array], )  # 对每一个值进行循环判断

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, _):
        return (tf.less(i, self.length + 1))  # 判断是否是最后一个值

    def body(self, i, _):
        flow = tf.cond(
            tf.equal(tf.mod(i, 15), 0),  # 如果值能被 15 整除，那么就把该位置赋值为 FizzBuzz
            lambda: tf.assign(self.array[i - 1], 'FizzBuzz'),

            lambda: tf.cond(tf.equal(tf.mod(i, 3), 0),  # 如果值能被 3 整除，那么就把该位置赋值为 Fizz
            lambda: tf.assign(self.array[i - 1], 'Fizz'),
            lambda: tf.cond(tf.equal(tf.mod(i, 5), 0),  # 如果值能被 5 整除，那么就把该位置赋值为 Buzz
                lambda: tf.assign(self.array[i - 1], 'Buzz'),
                lambda: self.array  # 最后返回的结果
                                            )
                            )
        )
        print("---------------")
        print("self.array:", self.array);
        return (tf.add(i, 1), flow)


if __name__ == '__main__':
    fizzbuzz = FizzBuzz(length=50)
    ix, array = fizzbuzz.run()
    print(array)

