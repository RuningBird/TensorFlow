import tensorflow as tf
import numpy as np

##########--------------------计数器-------------------##########
state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)  # 赋值

##如果设置了变量，这一步必须执行
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)  ##传进最后的表达式
        print(sess.run(state))

    print('finall=>',sess.run(state))
