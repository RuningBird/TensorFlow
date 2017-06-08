import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# mylayer = add_layer(xs, 784, 10, activation_function=tf.nn.relu)
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    print('',y_pre[0],'\n row ',np.size(y_pre,0))
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    print('argmax-->',sess.run(tf.argmax(y_pre, 1)),'size',np.alen(sess.run(tf.argmax(v_ys, 1))))
    for each in sess.run(tf.arg_max(y_pre,1)):
        print(each,end=' ')

    print('\ncorrect_prediction---->',sess.run(correct_prediction),'   type:',type(sess.run(correct_prediction)))
    print(sess.run(tf.cast(correct_prediction, tf.float32)))
    for each in sess.run(correct_prediction):
        print(each,end=' ')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


for i in range(1):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # print(type(batch_ys))
    # print(batch_ys)
    # print(np.alen(batch_ys))
    # print(np.size(batch_xs,1))
    print(mnist.test.labels)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 1 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
