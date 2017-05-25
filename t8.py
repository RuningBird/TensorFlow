import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
################################--------------可视化结果------------------##################################################

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 设置行列
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 设置偏执不为零

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:  # 线性问题
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


###############--------建造神经网络------------====================

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)# 加噪点
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])#None表示不论给多少个例子都ok
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# 求和等，要加reduce, 在求平均
loss1 = tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])
loss = tf.reduce_mean(loss1)

##怎么学习，optimizer function
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 要做的事情是减少loss
init = tf.initialize_all_variables()

#######################################################################################3
with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()#连续输入
    plt.show()#会暂停


    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50:
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.1)



################################--------------可视化结果------------------##################################################