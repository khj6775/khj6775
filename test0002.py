# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np

# #1. 데이터
# x = np.array([1,2,3,4,5,6])
# y = np.array([1,2,3,4,5,6])

# #2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1)

# #4. 평가, 예측
# loss = model.evaluate(x, y)
# print("로스 : ", loss)
# result = model.predict([1,2,3,4,5,6,7])
# print(result)

from tensorflow.keras.datasets import mnist

# MNIST 데이터 로드
(X, y), (Xt, yt) = mnist.load_data()

# 레이블 출력
n_th = 100
for i in range(3):  # 100번째부터 102번째까지 확인
    print(f'Label for sample {n_th + i}:', y[n_th + i])
