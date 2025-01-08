import tensorflow as tf
tf.compat.v1.set_random_seed(777)
# from sklearn.datasets import
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

#2. 모델
# layer1 : model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random_normal([2, 10], name='weight1'))
b1 = tf.compat.v1.Variable(tf.zeros([10], name='bias1'))

# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
layer1 = tf.compat.v1.matmul(x, w1) + b1        # (N, 10)

#################### 드랍아웃 적용 ##################################
# layer2 : model.add(Dense(5, input_dim=10))
w2 = tf.compat.v1.Variable(tf.random_normal([10, 8], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([8], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2   #(N, 8)
layer2 = tf.compat.v1.nn.dropout(layer2, rate=0.3)

################### relu 적용 #####################################
# layer3 : model.add(Dense(3, input_dim=5))
w3 = tf.compat.v1.Variable(tf.random_normal([8, 3], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([3], name='bias3'))
layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)
# layer3 = tf.compat.v1.nn.selu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)
# layer3 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)
# layer3 = tf.compat.v1.nn.elu(tf.compat.v1.matmul(layer2, w3) + b3)   # (N, 3)

# layer4 : model.add(Dense(4, input_dim=3))
w4 = tf.compat.v1.Variable(tf.random_normal([3, 4], name='weight4'))
b4 = tf.compat.v1.Variable(tf.zeros([4], name='bias4'))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

# output_layer : model.add(Dense(1, activation='sigmoid'))
w5 = tf.compat.v1.Variable(tf.random_normal([4, 1], name='weight5'))
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias5'))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

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