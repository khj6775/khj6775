
import numpy as np
import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=6666)


x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

print(x_train.shape)    # (404, 13, 1, 1)
print(x_test.shape)     # (102, 13, 1, 1)

x_train = x_train/255.
x_test = x_test/255.

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', strides=1,padding='same', input_shape=(13, 1, 1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
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
model.add(Dense(1))

model.summary()

# model.save("./_save/keras28/keras28_1_save_model.h5")   # 상대경로
# model.save("c:/AI5/_save/keras32/keras32_1_save_model.h5")   # 절대경로


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


path ='./_save/keras39_cnn01/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k39_boston', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
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
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end - start , 2),"초")


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


# DNN --> CNN
# 로스 :  34.690208435058594
# r2스코어 :  0.6804278904005451
# 걸린시간 :  7.7 초