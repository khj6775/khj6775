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


# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.nn.log_softmax(hypothesis), axis=1))  # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=201
batch_size = 100
total_batch = int(len(x_train) / batch_size)
print(total_batch)  # 600

for step in range(epochs):

    avg_cost = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}

        cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], 
                                             feed_dict=feed_dict)
        avg_cost +=cost_val
    avg_cost /= total_batch

    if step % 10 == 0 :
        print(step, 'loss : ', avg_cost)
        

#4. 평가, 예측
print("==============================================")
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_predict[0], y_predict.shape)    # (10000, 10)

y_predict_arg = sess.run(tf.math.argmax(y_predict, 1))
print(y_predict_arg[0], y_predict_arg.shape)    # 7 (10000,)

y_test = np.argmax(y_test, 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict_arg, y_test)
print('acc :', acc)     # acc : 0.9227