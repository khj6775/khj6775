import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],]
)

y = np.array([4,5,6,7,8,9,10,])

print(x.shape, y.shape) # (7, 3) (7,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1],1)
print(x.shape)
# 3-D tensor with shape (batch_size, timesteps, features)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))   # 행무시 열우선 행7 뺌
model.add(SimpleRNN(units=10, activation='relu', input_length=3,input_dim=1))   # 가독성 떨어진다. 안쓸란다.

model.add(Dense(16,activation='relu'))     # RNN은 바로 Dense 로 연결이 가능하다. 
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=300, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x,y, 
          epochs=1000, 
          batch_size=16, 
          validation_split=0.2,
        #   callbacks=[es]
)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_pred = np.array([8,9,10]).reshape(1,3,1)  # [[[8],[9],[10]]]
y_pred = model.predict(x_pred)
# (3, )->(1,3,1)

print('[8,9,10]의 결과 : ', y_pred)

# SimpleRNN
# loss : 1.4632488500865293e-06
# [8,9,10]의 결과 :  [[11.000445]]

# GRU
# loss : 0.0003673420287668705
# [8,9,10]의 결과 :  [[10.961616]]