import tensorflow as tf
###############--------定义一个添加层的函数------------====================

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 设置行列
    biases = tf.Varable(tf.zeros([1, out_size]) + 0.1)  # 设置偏执不为零

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:#线性问题
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
