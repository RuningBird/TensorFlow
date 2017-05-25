###########################----placeholder---#############################################################
# 每一次从外界传入值代替他
import tensorflow as tf

# imput1 = tf.placeholder(tf.float32,[2,2])##可设置结构，两行两列结构
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as ses:
    r = ses.run(output,feed_dict={input1:[7.],input2:[2.0]})#可以只设置一个.表示小数
    print(r)
