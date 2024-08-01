# keras21_2_kaggle_bank copy

# https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# #0 replace data
# PATH = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

# train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

# print(train_csv['Geography'].value_counts())

# train_csv['Geography'] = train_csv['Geography'].replace('France', value = 1)
# train_csv['Geography'] = train_csv['Geography'].replace('Spain', value = 2)
# train_csv['Geography'] = train_csv['Geography'].replace('Germany', value = 3)

# train_csv['Gender'] = train_csv['Gender'].replace('Male', value = 1)
# train_csv['Gender'] = train_csv['Gender'].replace('Female', value = 2)

# train_csv.to_csv(PATH + "replaced_train.csv")

# #1 replace data
# PATH = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

# test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

# print(test_csv['Geography'].value_counts())

# test_csv['Geography'] = test_csv['Geography'].replace('France', value = 1)
# test_csv['Geography'] = test_csv['Geography'].replace('Spain', value = 2)
# test_csv['Geography'] = test_csv['Geography'].replace('Germany', value = 3)

# test_csv['Gender'] = test_csv['Gender'].replace('Male', value = 1)
# test_csv['Gender'] = test_csv['Gender'].replace('Female', value = 2)

# test_csv.to_csv(PATH + "replaced_test.csv")



#1. 데이터
path = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'replaced_train.csv', index_col=0)
print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'replaced_test.csv', index_col=0)
print(test_csv)     # [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [110023 rows x 1 columns]

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(submission_csv.shape) # (110023, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 결측치 없음

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
train_csv=train_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis = 1)
print(x)    # [165034 rows x 10 columns]

y = train_csv['Exited']
print(y.shape)      # (165034,)


print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=1186)

print(x_train.shape, y_train.shape) # (132027, 10)
print(x_test.shape, y_test.shape)   # (33007, 10)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', 
                 strides=1,padding='same', input_shape=(5,2,1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 40,
    restore_best_weights=True
)


import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras39_08/kaggle_bank/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k39_08', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=333,
                 validation_split=0.2,
                 callbacks=[es, mcp]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('로스 :', loss)
print("acc :", round(loss[1],3))

y_pred = model.predict(x_test)
print(y_pred[:50])
y_pred = np.round(y_pred)
print(y_pred[:50])

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print('걸린시간 : ', round(end - start, 2), "초")

print(test_csv.shape)

# y_submit = np.round(model.predict(test_csv))      # round 꼭 넣기
# print(y_submit)
# print(y_submit.shape)     

# #################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
# submission_csv['Exited'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)

# submission_csv.to_csv(path + "submission_0723_1223.csv")

print('로스 :', loss)
print("acc :", round(loss[1],3))

# 로스 : [0.32803651690483093, 0.8616657257080078]
# acc : 0.862

# 로스 : [0.3283868730068207, 0.8620595335960388]
# acc : 0.862

# cnn 변환
# acc_score : 0.8639379525555185
# 걸린시간 :  133.01 초
# (110023, 10)
# 로스 : [0.3244544267654419, 0.8639379739761353]
# acc : 0.864