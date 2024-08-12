import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

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
model.add(Bidirectional(LSTM(units=10,), input_shape=(3,1)))

# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))   # 행무시 열우선 행7 뺌
# model.add(LSTM(units=10, activation='relu', input_shape=(3,1)))   # 통상적으로 LSTM 많이쓴다.
# # model.add(GRU(units=10, activation='relu', input_shape=(3,1)))   

model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.summary()

# GRU
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 475
# Trainable params: 475
# Non-trainable params: 0
# _________________________________________________________________


# Bidirectional(GRU)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               780
#  l)


# =================================================================
# Total params: 935
# Trainable params: 935
# Non-trainable params: 0
# _________________________________________________________________


# SimpleRNN
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               240
#  l)

#  dense (Dense)               (None, 7)                 147

# =================================================================


# Bidirectional(SimpleRNN)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               960
#  l)


# =================================================================
# Total params: 1,115
# Trainable params: 1,115
# Non-trainable params: 0
# _________________________________________________________________


# LSTM
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 565
# Trainable params: 565
# Non-trainable params: 0
# _________________________________________________________________


# Bidirectional(LSTM)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               960
#  l)

#  dense (Dense)               (None, 7)                 147

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 1,115
# Trainable params: 1,115
# Non-trainable params: 0
# _________________________________________________________________


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=300, verbose=1,
#                    restore_best_weights=True,
#                    )

# model.fit(x,y, 
#           epochs=1000, 
#           batch_size=16, 
#           validation_split=0.2,
#         #   callbacks=[es]
# )

# #4. 평가, 예측
# results = model.evaluate(x,y)
# print('loss :', results)

# x_pred = np.array([8,9,10]).reshape(1,3,1)  # [[[8],[9],[10]]]
# y_pred = model.predict(x_pred)
# # (3, )->(1,3,1)

# print('[8,9,10]의 결과 : ', y_pred)

# # SimpleRNN
# # loss : 1.4632488500865293e-06
# # [8,9,10]의 결과 :  [[11.000445]]

# # GRU
# # loss : 0.0003673420287668705
# # [8,9,10]의 결과 :  [[10.961616]]