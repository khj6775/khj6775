# keras22_softmax3_fetch_covtype copy

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))

# one hot encoding
y_ohe = pd.get_dummies(y)
print(y_ohe.shape)  # (581012, 7)

print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.1, random_state=5353)


x_train = x_train.reshape(-1,9,6,1)
x_test = x_test.reshape(-1,9,6,1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', 
                 strides=1,padding='same', input_shape=(9,6,1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
# model.add(MaxPooling2D())
# model.add(MaxPooling2D(pool_size=3, padding='same'))  # 커널사이즈(3,3)
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu',strides=1,padding='same'))
# model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(128, (2,2), activation='relu',strides=1,padding='same')) 
# model.add(Conv2D(128, 2, activation='relu',strides=1,padding='same'))  # 커널사이즈 간단히 2로만 표현할수도 있다.
# model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu',strides=1,padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())

# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu', input_shape=(32,)))
model.add(Dropout(0.1))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras39/10_fetch_covtype/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k39_10', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

model.fit(x_train, y_train, epochs=1000, batch_size=500,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
# print('걸린 시간 :', round(end-start, 2), '초')

# loss : 0.13126742839813232
# acc : 0.9509
# r2 score : 0.864602581653606

#cnn 변환
# loss : 0.5852555632591248
# acc : 0.7439
# r2 score : 0.4545169811371788
# acc_score : 0.7331761385150253