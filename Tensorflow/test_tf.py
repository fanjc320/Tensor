import numpy as np
import tflearn

from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

from tflearn.data_utils import load_csv
data,labels = load_csv('titanic_dataset.csv',target_column=0,
                       categorical_labels=True,n_classes=2)


# 上面使用load_csv()函数从csv文件中读取数据，并转为python List。其中target_column参数用于表示我们的标签列id，该函数将返回一个元组：（data,labels）。


# 抛弃输入中的姓名以及船票号码字段，并将性别字段转为数值，0表示男性，1表示女性，

def preprocess(data,colums_to_ignore):
    for id in sorted(colums_to_ignore,reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data,dtype=np.float32)

 # 其中的pred为对于[dicaprio,winslet]预测得到的结果，对于其中某一个（比如dicaprio）进行预测的结果为[死亡概率，存活概率]，所以这里打印的是pred[i][1]。

to_ignore = [1,6]
data = preprocess(data,to_ignore)

net = tflearn.input_data(shape=[None,6])
net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net,32)
net = tflearn.fully_connected(net,2,activation='softmax')
net = tflearn.regression(net)

# 训练
model = tflearn.DNN(net)
model.fit(data,labels,n_epoch=10,batch_size=16,show_metric=True)

# 预测

# Let's create some data for DiCaprio and Winslet
dicaprio = [3,'Jack Dawson','male',19,0,0,'N/A',5.0000]
winslet = [1,'Rose DeWitt Bukater','female',17,1,2,'N/A',100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("winslet Surviving Rate:", pred[1][1])