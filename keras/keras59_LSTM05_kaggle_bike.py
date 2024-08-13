# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

# keras19_EarlyStopping5_kaggle_bike copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
#1. 데이터
path = 'C:\\AI5\\_data\\kaggle\\bike-sharing-demand\\'      # 절대경로
# path = 'C://AI5//_data//bike-sharing-demand//'      # 절대경로   다 가능
# path = 'C:/AI5/_data/bike-sharing-demand/'      # 절대경로
#  /  //  \  \\ 다 가능

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# train_dt = pd.DatetimeIndex(train_csv.index)

# train_csv['day'] = train_dt.day
# train_csv['month'] = train_dt.month
# train_csv['year'] = train_dt.year
# train_csv['hour'] = train_dt.hour
# train_csv['dow'] = train_dt.dayofweek

# test_dt = pd.DatetimeIndex(test_csv.index)

# test_csv['day'] = test_dt.day
# test_csv['month'] = test_dt.month
# test_csv['year'] = test_dt.year
# test_csv['hour'] = test_dt.hour
# test_csv['dow'] = test_dt.dayofweek


print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(sampleSubmission.shape)   # (6493, 1)
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count

print(train_csv.columns)       # 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe().T)   # describe 평균,중위값 등등 나타냄. 많이쓴다.

############### 결측치 확인 #################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())             # 전부 결측치 없음 확인
print(test_csv.isnull().sum())

############# x와 y를 분리 ########
x = train_csv.drop(['casual', 'registered','count'], axis=1)    # 대괄호 하나 = 리스트    두개 이상은 리스트
print(x)            # [10886 rows x 8 columns]

y = train_csv['count']
print(y.shape)      #(10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,
                                                    random_state = 68481)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()

model.add(LSTM(units=16, input_shape=(8,1)))

model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
es = EarlyStopping(                                               
    monitor = 'val_loss',                                              
    mode = 'min',
    patience = 20,
    restore_best_weights = True
)

import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras32_mcp/05_kaggle_bike/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k32_05', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs = 600, batch_size=50,
                 validation_split=0.25,
                 callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 :", r2)
print("걸린시간 :", round(end-start,2),"초")


# 로스 : 20711.58203125
# r2 스코어 : 0.32213358845494955
# 걸린시간 : 14.35 초

# LSTM
# 로스 : 21073.916015625
# r2 스코어 : 0.3102749857937537
# 걸린시간 : 40.6 초