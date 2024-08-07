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
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))   # 행무시 열우선 행7 뺌

model.add(Dense(16,activation='relu'))     # RNN은 바로 Dense 로 연결이 가능하다. 
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120

#  dense (Dense)               (None, 16)                176

#  dense_1 (Dense)             (None, 32)                544

#  dense_2 (Dense)             (None, 64)                2112

#  dense_3 (Dense)             (None, 64)                4160

#  dense_4 (Dense)             (None, 32)                2080

#  dense_5 (Dense)             (None, 16)                528

#  dense_6 (Dense)             (None, 8)                 136

#  dense_7 (Dense)             (None, 1)                 9

# =================================================================
# Total params: 9,865
# Trainable params: 9,865
# Non-trainable params: 0
# _________________________________________________________________

# 120개의 비밀을 찾아라

# 파라미터 갯수 = units * (units + bias + feature)
#               = units*units + units*bias +  units*feature

