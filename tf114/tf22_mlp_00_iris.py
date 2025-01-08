# 01_boston.py

# 02_california.py

# 03_diabetes.py

# 04_dacon_darrung.py

# 05_kaggle_bike.py

# 06_cancer.py

# 07_dacon_diabetes.py

# 08_kaggle_bank.py

# 09_wine.py

# 10_fetch_convtype.py

# 11_digits.py

# 12_kaggle_santender.py

# 13_kaggle_otto.py

# 14_mnist.py 

# 15_fashion.py

# 16_cifar10.py

# 17_cifar100.py

# 18_kaggle_cat_dog.py

# 19_horse.py

# 20_rps.py

# 21_jena.py


import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1004,
                                                    stratify=y
                                                    )
print(x_train.shape, y_train.shape)     # (112, 4) (112, 3)
print(x_test.shape, y_test.shape)       # (38, 4) (38, 3)



# x = tf.placeholder(tf.float32, shape=[None, 4])
# y = tf.placeholder(tf.float32, shape=[None, 3])

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3], name='weight', dtype=tf.float32))
# b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias', dtype=tf.float32))

#2. 모델
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w1 = tf.compat.v1.Variable(tf.random_normal([4, 3], name='weight1'))
b1 = tf.compat.v1.Variable(tf.zeros([3], name='bias1'))

# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
layer1 = tf.compat.v1.matmul(x, w1) + b1        # (N, 10)

#################### 드랍아웃 적용 ##################################
# layer2 : model.add(Dense(5, input_dim=10))
w2 = tf.compat.v1.Variable(tf.random_normal([3, 2], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([2], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2   #(N, 8)
layer2 = tf.compat.v1.nn.dropout(layer2, rate=0.3)

################### relu 적용 #####################################
# layer3 : model.add(Dense(3, input_dim=5))
w3 = tf.compat.v1.Variable(tf.random_normal([2, 1], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([1], name='bias3'))
layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)
# layer3 = tf.compat.v1.nn.selu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)
# layer3 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)
# layer3 = tf.compat.v1.nn.elu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)

# # layer4 : model.add(Dense(4, input_dim=3))
# w4 = tf.compat.v1.Variable(tf.random_normal([3, 4], name='weight4'))
# b4 = tf.compat.v1.Variable(tf.zeros([4], name='bias4'))
# layer4 = (tf.compat.v1.matmul(layer3, w4) + b4)

# # output_layer : model.add(Dense(1, activation='sigmoid'))
# w5 = tf.compat.v1.Variable(tf.random_normal([4, 1], name='weight5'))
# b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias5'))
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer2, w3) + b3)

# hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf. reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))  # categorical_crossentropy

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w3, b3], 
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
print('acc :', acc)     # acc : 0.9473684210526315

sess.close()
