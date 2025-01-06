import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

x = [1,2,3]
y = [1,2,3]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.squre(hypothesis - y))   # mse

w_history = []
loss_