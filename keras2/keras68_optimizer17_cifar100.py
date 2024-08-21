import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

#. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      # (50000, 32, 32, 3) (50000, 1) 


## 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))

### 원핫1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape)     
print(x_test.shape, y_test.shape)       

#2. 모델
model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', input_shape=(32, 32, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(Dropout(0.2))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(128, (2,2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())

# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1, verbose=0,
          batch_size=512,
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss[0]))

    y_predict = model.predict(x_test, verbose=0)
    acc = loss[1]
    print('lr : {0}, acc : {1}'.format(learning_rate, acc))


#     =================1. 기본출력 ========================
# lr : 0.1, 로스 :4.614956378936768
# lr : 0.1, acc : 0.009999999776482582
# =================1. 기본출력 ========================
# lr : 0.01, 로스 :4.605802059173584
# lr : 0.01, acc : 0.009999999776482582
# =================1. 기본출력 ========================
# lr : 0.005, 로스 :4.605567932128906
# lr : 0.005, acc : 0.009999999776482582
# =================1. 기본출력 ========================
# lr : 0.001, 로스 :4.605554103851318
# lr : 0.001, acc : 0.009999999776482582
# =================1. 기본출력 ========================
# lr : 0.0005, 로스 :4.605547904968262
# lr : 0.0005, acc : 0.009999999776482582
# =================1. 기본출력 ========================
# lr : 0.0001, 로스 :4.605549335479736
# lr : 0.0001, acc : 0.009999999776482582

