# keras23_kaggle2_otto copt

# 과제  acc=0.89이상 만들어서 제출 캡쳐 이메일로 전송
# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path = 'C:/AI5/_data/kaglle/Otto Group/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [61878 rows x 94 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [144368 rows x 93 columns]

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission_csv)   # [144368 rows x 9 columns]

print(train_csv.columns)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']
print(y.shape)

y_ohe = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,y_ohe,train_size=0.8,
                                                    random_state=3752,
                                                    stratify=y)


x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

print(x_train.shape, x_test.shape)


x_train = x_train.reshape(-1,31,3,1)
x_test = x_test.reshape(-1,31,3,1)

x_train = x_train/255.
x_test = x_test/255.



#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', 
                 strides=1,padding='same', input_shape=(31,3,1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
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
model.add(Dense(9, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 77,
    restore_best_weights=True
)


import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras39/13_kaggle_otto/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k39_13', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
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

accuracy_score = accuracy_score(y_test, y_pred)  
# r2 = r2_score(y_test, y_predict)
print("acc_score : ", accuracy_score)
print("걸린시간 : ", round(end - start , 2),"초")


y_submit = model.predict(test_csv.values.reshape(-1, 31, 3, 1))      # round 꼭 넣기
print(y_submit)
print(y_submit.shape)     

# #################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
# #submission_csv['target'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)

# submission_csv[['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']] = y_submit

# for i in range(9) :
#     submission_csv['Class_' + str(i+1)] = y_submit[:,i].astype('int')


# submission_csv.to_csv(path + "submission_0725_1620.csv")

print('로스 :', loss)
print("acc :", round(loss[1],3))

# 로스 : [0.5520421862602234, 0.7964608669281006]
# acc : 0.796

# Dropout 적용
# 로스 : [0.521433413028717, 0.7995313405990601]
# acc : 0.8
