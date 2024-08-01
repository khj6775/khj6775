import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

## 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))

# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)

### 원핫1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape)     
print(x_test.shape, y_test.shape)       

# #2. 모델
# model = Sequential()
# model.add(Dense(128,activation='relu', input_shape=(32*32*3,)))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))

# model.add(Dense(10, activation='softmax'))

# 함수형
input1 = Input(shape=(32,32,3))
conv1 = Conv2D(64, 3, activation='relu', 
               strides=1,padding='same', name='ys1')(input1)
maxpool1 = MaxPooling2D()(conv1)
batchnormal1 = BatchNormalization()(maxpool1)
conv2 = Conv2D(64, 3, activation='relu', 
               strides=1,padding='same', name='ys2')(batchnormal1)
maxpool2 = MaxPooling2D()(conv2)
drop1 = Dropout(0.3)(maxpool2)
conv3 = Conv2D(64, 2, activation='relu',
                strides=1,padding='same', name='ys3')(drop1)
maxpool3 = MaxPooling2D()(conv3)
drop2 = Dropout(0.2)(maxpool3)
conv4 = Conv2D(64, 2, activation='relu',
                strides=1,padding='same', name='ys4')(drop2)
drop3 = Dropout(0.2)(conv4)
conv5 = Conv2D(32, 2, activation='relu',
                strides=1,padding='same', name='ys5')(drop3)
drop4 = Dropout(0.1)(conv5)


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


path ='./_save/keras40_dnn3/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k40_03', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=512,
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


# MaxPooling 적용
# loss : 0.6229567527770996
# acc : 0.8
# accuracy_score : 0.7972
# 걸린 시간 : 384.29 초

# BatchNormalization 적용
# loss : 0.6133089661598206
# acc : 0.81
# accuracy_score : 0.8109
# 걸린 시간 : 673.22 초

# dnn
# loss : 1.6240884065628052
# acc : 0.42
# accuracy_score : 0.4176
# 걸린 시간 : 195.48 초

#함수형
# loss : 0.5933529734611511
# acc : 0.8
# accuracy_score : 0.7997
# 걸린 시간 : 286.03 초