# keras22_softmax3_fetch_covtype copy

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=54, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
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
model.fit(x_train, y_train, epochs=1000, batch_size=500,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
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
print('걸린 시간 :', round(end-start, 2), '초')


"""
random_state=5353
epochs=1000, batch_size=500
patience=60
loss : 0.20093396306037903
acc : 0.9228
r2 score : 0.7992267483422014
acc_score : 0.92077725379505
걸린 시간 : 466.28 초
"""

# MinMaxScaler 적용
# loss : 0.14127600193023682
# acc : 0.97
# r2 score : 0.9343391961683192
# acc_score : 0.9555555555555556

# StandardScaler
# loss : 0.13319317996501923
# acc : 0.9486
# r2 score : 0.849960627497632
# acc_score : 0.947884754397439

# MaxAbsScaler
# loss : 0.1533125638961792
# acc : 0.9402
# r2 score : 0.8347796292779198
# acc_score : 0.9390726653127259
# 걸린 시간 : 461.15 초

# RobustScaler
# loss : 0.13072569668293
# acc : 0.9514
# r2 score : 0.8596564469031069
# acc_score : 0.9509655433547899
# 걸린 시간 : 548.91 초