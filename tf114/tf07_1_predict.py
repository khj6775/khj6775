import tensorflow as tf

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])


# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


#2. 모델구성
# y = wx + b => y = xw + b  # 이제는 말할수 있다.
hypothesis = x * w + b

#3-1. 컴파일 model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit()
    epochs=500
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                 feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)
# sess.close()

    #4. 예측.
    print("================== predict ======================")
    ############################ [ 실습 ] ###################################
    # x_pred_data = [6, 7, 8]
    # # 예측값을 뽑아봐.


    # # placeholder 사용

    # x_test = tf.placeholder(tf.float32, shape=[None])
    # y_predict = tf.placeholder(tf.float32, shape=[None])
    # predict = sess.run([x_test, y_predict], feed_dict = {x_test:x_pred_data, y_predict:x_pred_data * w_val + b_val})
    # print(predict)

    # # placeholder 안쓰고
    # x_test = tf.constant(x_pred_data, dtype=tf.float32)
    # y_predict = x_test * w_val + b_val
    # print(sess.run(y_predict))


    x_test = [6, 7, 8]

    # 선생님이 알려주신거
    #1. 파이썬 방식
    y_predict = x_test * w_val + b_val
    print('[6,7,8]의 예측 :', y_predict)

    #2. placeholder에 넣어서
    x_test_ph = tf.compat.v1.placeholder(tf.float32, shape=[None])
    
    y_predict2 = x_test_ph * w_val + b_val
    print('[6,7,8]의 예측 :', sess.run(y_predict2, feed_dict={x_test_ph:x_test}))



