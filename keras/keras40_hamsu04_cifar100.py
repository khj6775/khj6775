import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, Input
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
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
input1 = Input(shape=(32,32,3))
conv1 = Conv2D(64, 3, activation='relu', 
               strides=1,padding='same', name='ys1')(input1)
maxpool1 = MaxPooling2D()(conv1)
batchnormal1 = BatchNormalization()(maxpool1)
drop1 = Dropout(0.2)(batchnormal1)
conv2 = Conv2D(128, 3, activation='relu', 
               strides=1,padding='same', name='ys2')(drop1)
maxpool2 = MaxPooling2D()(conv2)
batchnormal2 = BatchNormalization()(maxpool2)
drop2 = Dropout(0.2)(batchnormal2)
conv3 = Conv2D(128, 3, activation='relu', 
               strides=1,padding='same', name='ys3')(drop2)
maxpool3 = MaxPooling2D()(conv3)
batchnormal3 = BatchNormalization()(maxpool3)
drop3 = Dropout(0.2)(batchnormal3)
conv4 = Conv2D(128, 3, activation='relu', 
               strides=1,padding='same', name='ys4')(drop3)
drop4 = Dropout(0.2)(conv4)
batchnormal4 = BatchNormalization()(drop4)
drop5 = Dropout(0.1)(batchnormal4)

flat = Flatten()(drop5)

dense2 = Dense(units=128, activation='relu')(flat)
drop6 = Dropout(0.1)(dense2)
output1 = Dense(100, activation='softmax')(drop6)
model = Model(inputs=input1, outputs=output1) 


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='./_save/keras37_04/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k37_04', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
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

# MaxPooling 적용
# loss : 2.2045531272888184
# acc : 0.43
# accuracy_score : 0.428
# 걸린 시간 : 64.54 초

# loss : 2.0727145671844482
# acc : 0.47
# accuracy_score : 0.4699
# 걸린 시간 : 141.4 초

# val_accuracy, max 로 바꿈, BatchNormalization 추가
# loss : 2.6442418098449707
# acc : 0.53
# accuracy_score : 0.5252
# 걸린 시간 : 2130.92 초

# 함수형
# loss : 1.9940555095672607
# acc : 0.49
# accuracy_score : 0.4872
# 걸린 시간 : 99.48 초