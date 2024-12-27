# a03_ae2를 카피해서 모델 구성
# 인코더          28
#     conv        26
#     maxpool     13
#     conv        11
#     maxpool      5

# 디코더
#     conv                 7
#     UpSampling2D(2,2)   14
#     conv                14
#     UpSampling2D(2,2)   28
#     conv(1, (3,3))          -> (28,28,1) 로 맹그러


import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.
                                                    # 평균 0, 표편 0.1인 정규분포!!
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)        # shape 똑같다.
print(np.max(x_train), np.min(x_test))                  # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.506013411202829 -0.5281790150375157

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)

print(np.max(x_train_noised), np.min(x_test_noised))    # 1.0 0.0


#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):     # 레이어 조절을 편하게 하기 위해 함수형으로 변경
    model = Sequential()

    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', strides=1,padding='same',input_shape=(28, 28, 1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
    model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
    model.add(Dropout(0.1))

    model.summary()

            # 필터로 증폭, 커널 사이즈로 자른다.                              
                                    # shape = (batch_size, height, width, channels), (batch_size, rows, columns, channels)   
                                    # shape = (batch_size, new_height, new_width, filters)
                                    # batch_size 나누어서 훈련한다
    model.add(Flatten())

    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(units=16, activation='relu', input_shape=(32,)))
                            # shape = (batch_size, input_dim)
    model.add(Dense(10, activation='softmax'))

    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model


autoencoder = autoencoder(hidden_layer_size=hidden_size)


#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# relu, linear 랑은 별로. tanh도 별로.   sigmoid 랑 제일좋다.                       액티베이션과 로스 잘 생각해서 설정해줄것.

autoencoder.fit(x_train_noised, x_train, epochs=30, batch_size=128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)


import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()