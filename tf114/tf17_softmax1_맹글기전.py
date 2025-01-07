import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

#1. 데이터
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32))

#2. 모델
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf. reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], 
                           feed_dict={x:x_data, y:y_data})
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)
        # 2000 loss :  0.08405631

print(w_val, b_val)
# [[1.9226679]
#  [0.706727 ]] [-8.058993]

#4. 평가, 예측
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])

# y_predict = tf.sigmoid(x_test * w_val + b_val)
y_pred = tf.nn.softmax(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.argmax(y_pred, axis=1),
                     feed_dict={x_test:x_data})
print(y_predict)

y_true = np.argmax(y_data, axis=1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_predict)
print('acc : ', acc)


# 2000 loss :  0.44914365
# [[-5.1706033  -0.03973658  5.4743705 ]
#  [-0.28981066 -0.64860815  1.8383018 ]
#  [ 0.09100311  0.14143106 -2.2843032 ]
#  [ 2.2515242   1.8322232  -0.93976474]] [1.0818255e-06]
# [2 2 2 1 1 1 1 0]
# acc :  0.875