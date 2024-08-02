import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

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
model.add(Conv2D(64, (2,2), activation='relu', strides=1,padding='same', input_shape=(32, 32, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(BatchNormalization)
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',strides=1,padding='same'))

model.add(Dropout(0.2))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(128, (2,2), activation='relu',strides=1,padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu',strides=1,padding='same'))
model.add(Dropout(0.1))
model.add(Flatten())

# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu', input_shape=(32,)))
model.add(Dropout(0.1))

model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='./_save/keras36_04/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k36_04', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)


start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )

end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

# import tensorflow as tf
# (cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
# print(cifar_x.shape, cifar_y.shape)

# import matplotlib.pyplot as plt
# plt.imshow(cifar_y)

# loss : 2.884718656539917
# acc : 0.29
# accuracy_score : 0.2933
# 걸린 시간 : 278.64 초

# loss : 2.803560972213745
# acc : 0.3
# accuracy_score : 0.3021
# 걸린 시간 : 191.33 초