import tensorflow as tf

# indices = [1,2,3,4]
# with tf.Session() as sess:
#     x = tf.one_hot( indices =indices, depth = 5)
#     print(sess.run(x))
#
# x = tf.placeholder(tf.float32, [2, 3])
# with tf.Session() as sess:
#     print(sess.run(x,feed_dict={x:[[1,2,3],[2,2,3]]}))

# tensor = tf.zeros([3, 2])
# tensor1 = tf.ones([3,5])
# W = tf.Variable(tensor1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(W))


# with tf.InteractiveSession() as sess:
#     print(sess.run(W))

# a = tf.constant(2,shape=[5,2])
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(a)


# tensor1 = tf.zeros([2,5],name='test')
# with tf.Session() as sess:
#     print(sess.run(tensor1))
#     print(tensor1)
