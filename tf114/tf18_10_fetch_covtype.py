# 00_iris
# 09_wine
# 10_fetch_covtype
# 11_digits

# 맹그러봐!!!! # 다중분류

import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1004,
                                                    stratify=y
                                                    )
print(x_train.shape, y_train.shape)     # (435759, 54) (435759, 8)
print(x_test.shape, y_test.shape)       # (145253, 54) (145253, 8)

# exit()


x = tf.placeholder(tf.float32, shape=[None, 54])
y = tf.placeholder(tf.float32, shape=[None, 8])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([54,8], name='weight', dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

#2. 모델
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf. reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # categorical_crossentropy

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)


#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=501
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], 
                           feed_dict={x:x_train, y:y_train})
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)
        # 

print(w_val, b_val)
#

#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_predict)

y_predict = np.argmax(y_predict, 1)
print(y_predict)    # 
y_data = np.argmax(y_test, 1)
print(y_data)   # 

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)
print('acc :', acc)     # 

sess.close()
