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

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1, 3], name='bias'))

#2. 모델
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf. reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # categorical_crossentropy

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)


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
# [[-4.4116335   1.2211041   3.4545596 ]
#  [ 0.2394191  -0.07925612  0.7397169 ]
#  [ 0.45378232 -0.06332941 -2.4423194 ]
#  [ 1.8595449   1.2477084   0.03673127]] [[-4.0499887 -1.0651275  5.115123 ]]

#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_data})
print(y_predict)
# [[1.6503107e-06 1.5597227e-03 9.9843866e-01]
#  [1.8463197e-06 1.2929440e-01 8.7070376e-01]
#  [2.6414671e-08 1.5203728e-01 8.4796268e-01]
#  [2.8460432e-09 8.8254875e-01 1.1745122e-01]
#  [3.1773859e-01 6.6738105e-01 1.4880359e-02]
#  [1.5138471e-01 8.4852141e-01 9.3886840e-05]
#  [5.1696092e-01 4.8290807e-01 1.3101625e-04]
#  [7.3075771e-01 2.6919299e-01 4.9349102e-05]]
y_predict = np.argmax(y_predict, 1)
print(y_predict)    # [2 2 2 1 1 1 0 0]
y_data = np.argmax(y_data, 1)
print(y_data)   # [2 2 2 1 1 1 0 0]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)
print('acc :', acc)     # acc : 1.0

sess.close()
