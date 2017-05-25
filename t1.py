import tensorflow as tf
import numpy as np

# 创建data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### 创建tensorflow 结构  start ###
# 一维矩阵，随机范围（-1，1）
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

#定义loss函数
loss = tf.reduce_mean(tf.square(y - y_data))

#建立优化器，学习率0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化神经网络结构
init = tf.initialize_all_variables()
### 创建tensorflow 结构  srop ###

####  运行tensorflow ###
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights),sess.run(biases))






