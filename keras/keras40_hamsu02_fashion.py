
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

## 스케일링
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

print(np.max(x_train), np.min(x_train))

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

### 원핫
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape)     # (60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape)       # (10000, 28, 28, 1) (10000, 10)

# #2. 모델
# model = Sequential()
# model.add(Dense(128,activation='relu', input_shape=(28*28,)))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 함수형
input1 = Input(shape=(28,28,1))
conv1 = Conv2D(64, 3, activation='relu', 
               strides=1,padding='same', name='ys1')(input1)
maxpool1 = MaxPooling2D()(conv1)
drop1 = Dropout(0.3)(maxpool1)
conv2 = Conv2D(64, 3, activation='relu', 
               strides=1,padding='same', name='ys2')(drop1)
drop2 = Dropout(0.3)(conv2)
maxpool2 = MaxPooling2D()(drop2)
conv3 = Conv2D(64, 2, activation='relu',
                strides=1,padding='same', name='ys3')(maxpool2)
drop3 = Dropout(0.2)(conv3)
maxpool3 = MaxPooling2D()(drop3)
conv4 = Conv2D(64, 2, activation='relu',
                strides=1,padding='same', name='ys4')(maxpool3)
drop4 = Dropout(0.1)(conv4)
flat = Flatten()(drop4)
dense1 = Dense(units=32, activation='relu')(flat)
drop4 = Dropout(0.1)(dense1)
dense2 = Dense(units=16, activation='relu', input_shape=(32,))(drop4)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='./_save/keras40_dnn2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k40_02', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1024,
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

# loss : 0.2865922152996063
# acc : 0.91
# accuracy_score : 0.9095
# 걸린 시간 : 395.8 초

# MaxPooling 적용
# loss : 0.2029230147600174
# acc : 0.93
# accuracy_score : 0.9274
# 걸린 시간 : 301.09 초

# dnn
# loss : 0.3476593494415283
# acc : 0.88
# accuracy_score : 0.8809
# 걸린 시간 : 35.54 초

# 함수형
# loss : 0.3417896330356598
# acc : 0.88
# accuracy_score : 0.882
# 걸린 시간 : 40.65 초