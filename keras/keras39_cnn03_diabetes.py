# keras19_EarlyStopping3_diabetes copy

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
import time
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

#[실습] 맹그러봐
# R2 0.62 이상

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state= 8000)

x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', strides=1,padding='same', input_shape=(5, 2, 1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
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


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=50,
    restore_best_weights=True,
)

import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras39_cnn03/diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k39_03', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, validation_split=0.2,
           epochs=1000, batch_size=32,
           callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어: ", r2)
print("걸린시간 :", round(end-start, 2), "초")

# 로스 :  2517.196044921875
# r2 스코어:  0.6161683899501937
# 걸린시간 : 4.83 초

# CNN 변환
# 로스 :  3430.725341796875
# r2 스코어:  0.47686993417824575
# 걸린시간 : 10.36 초