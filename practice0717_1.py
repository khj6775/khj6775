# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
# path = 'C:\AI5\_data\bike-sharing-demand'
path = 'C://AI5//_data//bike-sharing-demand//'
# path = 'C:\\AI5\\_data\\bike-sharing-demand\\'
# path = 'C:/AI5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(sampleSubmission.shape)   # (6493, 1)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe().T)

############## 결측치 확인 ##############
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())

############## x와 y를 분리 #############
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # 두개이상 리스트
print(x)

y = train_csv['count']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,
                                                    random_state = 75614)
#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer="adam")
model.fit(x_train, y_train, epochs = 1000, batch_size=250)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어:", r2)

# y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)

# sampleSubmission['count'] = y_submit
# print(sampleSubmission)
# print(sampleSubmission.shape)

# sampleSubmission.to_csv(path + "sampleSubmission_0717_2101.csv")












