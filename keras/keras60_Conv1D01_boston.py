# 과적합이 문제다

# keras29_MCP5 copy

import numpy as np
import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # DESCR = pandas 의 describe
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      #(506, 13)    --> input_dim=13
print(y)
print(y.shape)      #(506,)  벡터

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# print('x_test :', x_test)
# print('y_train :', y_train)
# print('y_test :', y_test)


#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=13))

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(13, 1)))
model.add(Conv1D(10, 2))

model.add(Flatten())

model.add(Dense(32, input_shape=(13,)))   # input_shape 는 벡터형태로  # 이미지 input_shape=(8,8,1)
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))                    # 0.3 = 30퍼센트 빼고 훈련할거에요
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))

model.summary()

# model.save("./_save/keras28/keras28_1_save_model.h5")   # 상대경로
model.save("c:/AI5/_save/keras32/keras32_1_save_model.h5")   # 절대경로


# 그 모델의 가장 성능이 좋은 지점을 저장한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True
                   )
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='./_save/keras59_boston/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k59_', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
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
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end - start , 2),"초")
print(y_predict.shape, y_test.shape)
# print("============================= hist =========================")
# print(hist)
# print("============================= hist.history ==================")
# print(hist.history)
# print("============================= loss ==================")
# print(hist.history['loss'])
# print("============================= val_loss ==================")
# print(hist.history['val_loss'])
# print("==========================================================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.figure(figsize=(9,6))       # 그림판 사이즈
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('보스턴 loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()      # 모눈종이
# plt.show()


# {'loss': [51104.89453125, 4791.69677734375, 2480.6640625, 1702.1485595703125, 1233.7120361328125, 979.5753173828125, 786.275390625, 628.3425903320312, 506.5628662109375, 412.4776306152344], 'val_loss': [2691.3779296875, 4239.00390625, 1768.7796630859375, 1355.718017578125, 1018.6663208007812, 
# 800.6229858398438, 631.0289306640625, 499.1551513671875, 400.5531921386719, 327.1961975097656]}  =  딕셔너리

# (x, y, train_size=0.8, random_state=3333)  epochs=400, batch_size=5
# 로스 :  19.67884063720703
# r2스코어 :  0.7459871274196113
# (x, y, train_size=0.8, random_state=6666)  epochs=400, batch_size=5
# 로스 :  20.507410049438477
# r2스코어 :  0.7809722401357289
# (x, y, train_size=0.8, random_state=6666)  epochs=500, batch_size=5
# 로스 :  20.05512809753418
# r2스코어 :  0.7858027954982522

print('로스 : ', loss)

# St scaler 로스 :  19.375280380249023
# MM scaler 로스 :  20.29813003540039
# MAxAbsScaler 로스 : 18.711856842041016
# RobustScaler 로스 :  18.9057559967041


# ModelCheckpoint
# 로스 :  8.478630065917969
# r2스코어 :  0.9318442053653067
# 걸린시간 :  2.54 초

# 로스 :  11.00588607788086
# r2스코어 :  0.9115287661927284
# 걸린시간 :  2.33 초

# LSTM
# 로스 :  48.104270935058594
# r2스코어 :  0.6133119481876692
# 걸린시간 :  8.14 초


# Conv1D
# 로스 :  27.473526000976562
# r2스코어 :  0.7791530512111136
# 걸린시간 :  1.96 초