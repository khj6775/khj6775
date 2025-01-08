import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
tf.set_random_seed(333)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#################[실습] 맹그러봐 #############################

#2. 모델
# layer1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[784, 128],
                               initializer=tf.contrib.layers.xavier_initializer())  # kernel initializer
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')
layer1 = tf.compat.v1.matmul(x, w1) + b1
layer1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

w2 = tf.compat.v1.get_variable('w2', shape=[128, 64],
                               initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2
layer2 = tf.compat.v1.nn.relu(layer2)
layer2 = tf.compat.v1.nn.dropout(layer2, rate=0.3)

w3 = tf.compat.v1.get_variable('w3', shape=[64, 32],
                               initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.zeros([32]), name='b3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3
layer3 = tf.compat.v1.nn.relu(layer3)
layer3 = tf.compat.v1.nn.dropout(layer3, rate=0.3)

w4 = tf.compat.v1.get_variable('w4', shape=[32, 10])
b4 = tf.compat.v1.Variable(tf.zeros([10]), name='b4')
layer4 = tf.compat.v1.matmul(layer3, w4) + b4
hypothesis = tf.compat.v1.nn.softmax(layer4)



# w1 = tf.compat.v1.Variable(tf.random_normal([784, 256], name='weight1'))
# b1 = tf.compat.v1.Variable(tf.zeros([256], name='bias1'))

# # hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# layer1 = tf.compat.v1.matmul(x, w1) + b1        # (N, 10)

# # layer2 : model.add(Dense(5, input_dim=10))
# w2 = tf.compat.v1.Variable(tf.random_normal([256, 128], name='weight2'))
# b2 = tf.compat.v1.Variable(tf.zeros([128], name='bias2'))
# layer2 = tf.nn.softmax(tf.compat.v1.matmul(layer1, w2) + b2)

# # layer3 : model.add(Dense(3, input_dim=5))
# w3 = tf.compat.v1.Variable(tf.random_normal([128, 64], name='weight3'))
# b3 = tf.compat.v1.Variable(tf.zeros([64], name='bias3'))
# layer3 = tf.nn.softmax(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)

# # layer4 : model.add(Dense(4, input_dim=3))
# w4 = tf.compat.v1.Variable(tf.random_normal([64, 32], name='weight4'))
# b4 = tf.compat.v1.Variable(tf.zeros([32], name='bias4'))
# layer4 = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4) + b4)

# # output_layer : model.add(Dense(1, activation='sigmoid'))
# w5 = tf.compat.v1.Variable(tf.random_normal([32, 10], name='weight5'))
# b5 = tf.compat.v1.Variable(tf.zeros([10], name='bias5'))
# # hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)
# hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)


# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.nn.log_softmax(hypothesis), axis=1))  # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# epochs=201
# for step in range(epochs):
#     cost_val, _, = sess.run([loss, train], 
#                            feed_dict={x:x_train, y:y_train})
#     if step % 20 == 0 :
#         print(step, 'loss : ', cost_val)
#         #

epochs=401
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], 
                           feed_dict={x:x_train, y:y_train})
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)
        # 

# print(w_val, b_val)
   
# hypo, pred, acc = sess.run([hypothesis, predicted, accuracy], 
#                             feed_dict={x:x_test, y:y_test})

# print("훈련값 : ", hypo)
# print("예측값 : ", pred)
# print("acc : ", acc)

#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_predict[0], y_predict.shape)    # (10000, 10)

y_predict_arg = sess.run(tf.math.argmax(y_predict, 1))
print(y_predict_arg[0], y_predict_arg.shape)    # 7 (10000,)

y_test = np.argmax(y_test, 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict_arg, y_test)
print('acc :', acc)     # acc : 0.9168


# # print(y_predict)    # 
# y_data = np.argmax(y_test, 1)
# # print(y_data)   # 

# sess.close()