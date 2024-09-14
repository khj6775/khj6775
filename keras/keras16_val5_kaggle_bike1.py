# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = 'C:/AI5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)
print(test_csv.shape)
print(sampleSubmission.shape)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe().T)

############### 결측치 확인 ###########
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

############ x와 y를 분리 ###########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)

y = train_csv[['casual', 'registered']]
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.95,
                                                    random_state=7979)
print('x_train :', x_train.shape)
print('x_test :', x_test.shape)
print('y_train :', y_train.shape)
print('y_test :', y_test.shape )

#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(2, activation='linear'))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500,
          validation_split=0.2, batch_size=333)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 :", r2)

print(test_csv.shape)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

print("test_csv타입 :", type(test_csv))
print("y_submit타입:", type(y_submit))

test2_csv = test_csv
print(test2_csv.shape)

test2_csv[['casual', 'registered']] = y_submit
print(test2_csv)

test2_csv.to_csv(path + "test2.csv")


print('로스 :', loss)
print("r2 스코어 :", r2)

# 로스 : 10803.9482421875
# r2 스코어 : 0.295058239733956
