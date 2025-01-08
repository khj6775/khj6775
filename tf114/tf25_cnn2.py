import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_train = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]])
print(x_train.shape)    # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])

w = tf.compat.v1.constant([[[[1.]], [[0.]]],
                           [[[1.]], [[0.]]]])
print(w)    # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 1])
                                    #커널사이즈=(2,2), 컬러(채널)=1, 필터(아웃풋)=64
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')
                        # strides=2 일 경우? [1,2,2,1]
print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)   # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)

# # Layer2
# w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 32])
#                                     #커널사이즈=(3,3), 컬러(채널)=64, 필터(아웃풋)=32
# L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='VALID')
# # L2 = tf.nn.conv2d(L1, w2, strides=[1,2,2,1], padding='VALID')
#                         # strides=2 일 경우? [1,2,2,1]
# print(w2)   # <tf.Variable 'w2:0' shape=(3, 3, 64, 32) dtype=float32_ref>
# print(L2)   # Tensor("Conv2D_1:0", shape=(?, 25, 25, 32), dtype=float32)