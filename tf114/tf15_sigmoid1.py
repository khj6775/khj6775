import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]  # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]]              # (6,1)

##############################################
### [실습] 기냥 한번 맹그러봐!!!
##############################################

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32))

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

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
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

# y_predict = tf.sigmoid(x_test * w_val + b_val)
y_pred = tf.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.cast(y_pred >0.5, dtype=tf.float32),
                     feed_dict={x_test:x_data})
print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)    # acc :  1.0