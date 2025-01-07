import tensorflow as tf
tf.set_random_seed(1188)

#1. 데이터
x_data = [[73, 51, 65],                         # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]   
y_data = [[152], [185], [180], [205], [142]]     # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bias'))

#### [실습 맹그러봐!!!] ~~~~~~ ###########

#2. 모델
# hypothesis = x1*w1 + x2*w2 + x3*w3 + b
# hypothesis = x*w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00004)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=5001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], 
                           feed_dict={x:x_data, y:y_data})
    if step %  100 == 0 :
        print(step, 'loss : ', cost_val)

sess.close()


# 5000 loss :  284.64914