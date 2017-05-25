import tensorflow as tf
import numpy as np

### 矩阵相乘

matrix1 = tf.constant([[3, 3]])  # 一行两列
matrix2 = tf.constant([
    [2],
    [2]
])  # 两行一列

product = tf.matmul(matrix1, matrix2)  # 矩阵乘法，np.dot(m1, m2)

### ---------------- method 1 ----------------###
# sess = tf.Session()
# result = sess.run(product)
#
# print((result))
#
# sess.close()#可以没有

### ---------------- method 2----------------###
with tf.Session() as sess:
    res2 = sess.run(product)
    print(res2)