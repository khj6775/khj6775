# keras20_sigmoid_matrics_cancer copy

import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (569, 30) (569,)

print(np.unique(y, return_counts=True))     
# (array([0, 1]), array([212, 357], dtype=int64))  불균형 데이터인지 확인
print(type(x))  # <class 'numpy.ndarray'>  넘파이 파일

# print(y.value_counts())  # 에러
print(pd.DataFrame(y).value_counts())   # 넘파이를 판다스 데이터프레임으로 바꿔줘
# 1    357
# 0    212
print(pd.Series(y).value_counts())
pd.value_counts(y)          # 셋 다 똑같다.



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=315)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


x_train = x_train.reshape(-1,6,5,1)
x_test = x_test.reshape(-1,6,5,1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', strides=1,padding='same', input_shape=(6,5,1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
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
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])  # 매트릭스에 애큐러시를 넣으면 반올림해준다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras39_cnn06/cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k39_06', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=20,
                 validation_split=0.2,
                 callbacks=[es, mcp]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test,
                      verbose=1)
print('로스 : ', loss[0])
print("acc : ", round(loss[1],3))  # 애큐러시, 3자리 반올림 

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)  
# r2 = r2_score(y_test, y_predict)
print("acc_score : ", accuracy_score)
print("걸린시간 : ", round(end - start , 2),"초")

print('로스 : ', loss[0])
print("acc : ", round(loss[1],3))  # 애큐러시, 3자리 반올림 

# 걸린시간 :  1.58 초
# 로스 :  0.014798094518482685
# acc :  1.0

# CNN 변환
# acc_score :  0.9473684210526315
# 걸린시간 :  12.45 초
# 로스 :  0.14999598264694214
# acc :  0.947