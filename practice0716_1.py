# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "./_data/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)

print(submission_csv.shape)
print(train_csv.shape)
print(test_csv.shape)

print(train_csv.columns)

train_csv.info()

#### 결측치 삭제 #####
print(train_csv.isna().sum())

train_csv = train_csv.dropna()
print(train_csv.isna().sum())
print(train_csv)
print(train_csv.isna().sum())
train_csv.info()

test_csv.info()

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, random_state=4343)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs = 400, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)      #(715, 1)

######## submission.csv 만들기 ##################
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0716_1752.csv")