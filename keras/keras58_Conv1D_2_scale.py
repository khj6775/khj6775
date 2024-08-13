import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, Flatten

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print(x.shape, y.shape) # (7, 3) (7,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1],1)
print(x.shape)
# 3-D tensor with shape (batch_size, timesteps, features)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))   # 행무시 열우선 행7 뺌
# model.add(Bidirectional(LSTM(units=16), input_shape=(3,1)))   # 통상적으로 LSTM 많이쓴다.
# model.add(GRU(units=10, activation='relu', input_shape=(3,1)))   

# 데이터가 커질수록 성능이 좋아진다.

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1)))
model.add(Conv1D(10,2))

model.add(Flatten())

model.add(Dense(32,activation='relu'))     # RNN은 바로 Dense 로 연결이 가능하다. 
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# model = load_model('./_save/keras52/LSTM/k52_0807_1737_0202-0.0059.hdf5')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', mode='min', 
                   patience=200, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras56/Bidirectional/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k56_', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)


# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,  
#     save_best_only=True,  
#     filepath ="C:\\AI5\\_save\\keras52\\LSTM\\keras52.h5"       # 태운님이 알려준거
# )

model.fit(x,y, 
          epochs=2000, 
          batch_size=1, 
          callbacks=[es, mcp]
)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_pred = np.array([50,60,70]).reshape(1,3,1)
y_pred = model.predict(x_pred)
# (3, )->(1,3,1)

print('[50,60,70]의 결과 : ', y_pred)


# x_pred = np.array([50,60,70])

# loss : 0.0005832071183249354
# [50,60,70]의 결과 :  [[80.69334]]

# loss : 0.002362962579354644
# [50,60,70]의 결과 :  [[80.10468]]

# 17:20
# loss : [0.017035122960805893, 0.0]
# [50,60,70]의 결과 :  [[80.40093]]

# k52_0807_1722_0106-0.0628.hdf5
# loss : 0.20735175907611847
# [50,60,70]의 결과 :  [[79.7879]]

# Conv1D
# loss : 0.0046136206947267056
# [50,60,70]의 결과 :  [[80.03306]]

# loss : 0.0003029211948160082
# [50,60,70]의 결과 :  [[79.96773]]

# loss : 0.0001318450813414529
# [50,60,70]의 결과 :  [[80.036446]]