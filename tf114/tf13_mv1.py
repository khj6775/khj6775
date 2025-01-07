import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터
x1_data = [73., 93., 89., 96., 73.]         # 국어
x2_data = [80., 88., 91., 98., 66.]         # 영어
x3_data = [75., 93., 90., 100., 70.]        # 수학
y_data = [152., 185., 180., 196., 142.]     # 환산점수

# [실습] 맹그러봐!!!

# y = x1*w1 + x2*w2 + x3*w3 + b

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32))
b = tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b    # 핵심포인트

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))    # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00002)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=10001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data}
                           )
    if step % 100 == 0 :
        print(step, 'loss : ', cost_val)

sess.close()

# 10000 loss :  0.18931304