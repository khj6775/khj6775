import tensorflow as tf

#1. 데이터
x = [1,2,3]
y = [1,2,3]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


#2. 모델구성
# y = wx + b => y = xw + b  # 이제는 말할수 있다.
hypothesis = x * w + b

#3-1. 컴파일 model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit()
    epochs=500
    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))
# sess.close()    # close 안하면 메모리 많이 차지

# with 를 사용하면 sess.close 를 안해도 된다!

