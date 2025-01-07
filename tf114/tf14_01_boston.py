# boston

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
# from sklearn.datasets import
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# [실습] 맹그러봐
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 1], name='weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1] ,name='bias'))

#2. 모델
# hypothesis = x1*w1 + x2*w2 + x3*w3 + b
# hypothesis = x*w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

#3-2 훈련
w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import r2_score, mean_absolute_error

#3-2 훈련
for step in range(31):
    _, loss_v, w_v = sess.run([train, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v )

    w_history.append(w_v)
    loss_history.append(loss_v)

y_predict = tf.compat.v1.matmul(tf.cast(x_test, tf.float32), w_v)

y_predict1 = sess.run(y_predict)
print(y_predict1)

sess.close()

r2 = r2_score(y_test, y_predict1)
print("r2 스코어: ", r2)

