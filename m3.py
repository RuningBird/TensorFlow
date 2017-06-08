import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], 'input')
y = tf.placeholder(tf.float32, [None, 10], 'output')
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))

prediction = tf.nn.softmax(tf.matmul(x, w) + b)

#训练参数设置
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), 1))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

# def compute_accuracy(y, y_):
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys})
    return result

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    vx = mnist.test.images
    vy = mnist.test.labels
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        #先训练一块
        sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        if i % 20 == 0:#训练100次测试
            y_pre = sess.run(prediction,feed_dict={x:vx,y:vy})
            correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(vy,1))
            acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            # correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(mnist.test, 1))
            # ac = tf.reduce_mean()
            print(sess.run(acc),' cross_entropy',sess.run(cross_entropy,feed_dict={x:vx,y:vy}))



