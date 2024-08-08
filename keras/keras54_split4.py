# 54_3 카피해서
# (N,10,1) -> (n,5,2)
# 맹그러봐

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array(range(1,101))
x_predict = np.array(range(96, 106))    # 101부터 107을 찾아라

# 맹그러봐!!!

size = 11

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)      # append = 리스트 뒤에 붙인다.
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   

x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape)     

x = x.reshape(90,5,2)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=3752,
)

# exit()
#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))   # 행무시 열우선 행7 뺌
model.add(LSTM(units=16, activation='relu', input_shape=(5,2),)) #return_sequences=True))   # 통상적으로 LSTM 많이쓴다.
# model.add(LSTM(32))
# model.add(GRU(units=10, activation='relu', input_shape=(3,1)))   

# 데이터가 커질수록 성능이 좋아진다.

model.add(Dense(32,activation='relu'))     # RNN은 바로 Dense 로 연결이 가능하다. 
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# model.summary()


# 함수형
# inp_layer1 = Input(shape=(1,))
# m1 = Dense(1) (inp_layer1)

# inp_layer2 = Input(shape=(1,))
# m2 = Dense(1) (inp_layer2)

# m3 = concatenate([m1,m2])
# model = Model(inputs=[inp_layer1,inp_layer2],outputs=m3)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', mode='min', 
                   patience=80, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='C:/AI5/_save/keras54/split4/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k54_', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
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
          epochs=1000, 
          batch_size=8, 
          callbacks=[es, mcp]
)

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss :', results)

x_predict = x_predict.reshape(1,5,2)
y_pred = model.predict(x_predict)

print('107 나와라 : ', y_pred)

# loss : 0.00035284244222566485
# 107 나와라 :  [[106.16804]]

# loss : 0.0011585131287574768
# 107 나와라 :  [[106.036835]]

# shape (1,5,2)
# loss : 0.0006129173561930656
# 107 나와라 :  [[105.99752]]

# loss : 0.0007516654441133142
# 107 나와라 :  [[106.04515]]