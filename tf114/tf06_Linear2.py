import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터
x = [1,2,3,4,5]
y = [3,5,7,9,11]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

###[실습] 맹그러
# lr, epochs 만 조절


#2. 모델구성
# y = wx + b => xw + b
hypothesis = x * w + b

#3-1. 컴파일 model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# model.fit()
epochs = 500
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()

