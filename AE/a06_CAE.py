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

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, UpSampling2D



import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (2,2),input_shape=(28,28, 1), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(1, (3,3), padding='same'))
    return model

# hidden_size = 713       # pca 1.0
# hidden_size = 486       # pca 0.999
# hidden_size = 331       # pca 0.99
hidden_size = 154       # pca 0.95


model = autoencoder(hidden_layer_size=hidden_size)    

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss ='mse')
# autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')
model.fit(x_train_noised, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 5개 랜덤
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 원본 이미지 맨위에 그리기 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더기 츨략힌 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()