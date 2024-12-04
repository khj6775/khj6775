# keras28_1_save_model copy

import numpy as np
import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)   # DESCR = pandas 의 describe
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      #(506, 13)    --> input_dim=13
print(y)
print(y.shape)      #(506,)  벡터

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state=6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# print('x_test :', x_test)
# print('y_train :', y_train)
# print('y_test :', y_test)


# #2. 모델구성
# model = Sequential()
# # model.add(Dense(10, input_dim=13))
# model.add(Dense(32, input_shape=(13,)))   # input_shape 는 벡터형태로  # 이미지 input_shape=(8,8,1)
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

# model.summary()

# # model.save("./_save/keras28/keras28_1_save_model.h5")   # 상대경로
# model.save("c:/AI5/_save/keras28/keras28_1_save_model.h5")   # 절대경로


# 그 모델의 가장 성능이 좋은 지점을 저장한다.

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True
#                    )
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath = './_save/keras29_mcp/keras29_mcp1.hdf5'
# )


# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
#           verbose=1,
#           validation_split=0.2,
#           callbacks=[es, mcp]
#           )
# end = time.time

print("======================= 2. MCP 출력 ========================")
model = load_model('./_save/keras30_mcp/01_boston/k30_0727_2259_0099-12.6599.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print('로스 : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)


print('로스 : ', loss)

# 로스 :  6.734263896942139
# r2스코어 :  0.945866351588400