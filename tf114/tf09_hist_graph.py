# tf07_1_predict_선생님.py 카피

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

#2. 모델 구성
# y = wx + b => y = xw + b  # 이제는 말할 수 있다.

hypothesis = x * w + b

# 3-1. 컴파일 
# model.compoile (loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   # <- Adam쓰고 싶으면 GradientDescentOptimizer을 Adam으로 바꾸면 됨.
train = optimizer.minimize(loss)

# 3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # 변수 초기화

    # model.fit()
    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                               feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
        #     print(step, sess.run(loss), sess.run(w), sess.run(b))   # <- verbose임.
              print(step, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
# sess.close()   

    # 4. 예측
    print("================= predict ==================")
    ################## [실습] ##################################
    x_test = [6,7,8]
    # 예측값 뽑아봐.
    # y_predict = xw + b
    
    #1. 파이썬 방식
    y_predict = x_test * w_val 
    print('[6,7,8]의 예측: ', y_predict)    # [6,7,8]의 예측:  [12.01092196 14.01274228 16.01456261]
    
    #2. placeholder에 넣어서
    x_test_ph = tf.compat.v1.placeholder(tf.float32, shape=[None])
    
    y_predict2 = x_test_ph * w_val + b_val
    print('[6,7,8]의 예측: ', sess.run(y_predict2, feed_dict={x_test_ph: x_test}))    # [6,7,8]의 예측:  [13.004351 15.006171 17.007992]
    
print("===================== 그림그리기 =========================")
print(loss_val_list)
print(w_val_list)

# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# plt.plot(w_val_list)
# plt.xlabel('epochs')
# plt.ylabel('weighs')
# plt.grid()
# plt.show()

# plt.plot(w_val_list, loss_val_list)
# plt.xlabel('weighs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# subplot으로 위 3개의 그래프를 1개로 그려!!!

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# 첫 번째 그래프: Loss vs. Epochs
axes[0].plot(loss_val_list)
axes[0].set_title("Loss vs. Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].grid()

# 두 번째 그래프: Weights vs. Epochs
axes[1].plot(w_val_list)
axes[1].set_title("Weights vs. Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Weights")
axes[1].grid()

# 세 번째 그래프: Loss vs. Weights
axes[2].plot(w_val_list, loss_val_list)
axes[2].set_title("Loss vs. Weights")
axes[2].set_xlabel("Weights")
axes[2].set_ylabel("Loss")
axes[2].grid()

# 레이아웃 조정 및 표시
plt.tight_layout()
plt.show()