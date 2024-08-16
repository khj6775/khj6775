from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이타
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 소수 네째자리까지 맞추면 합격. 예: 6.0000 또는 5.9999

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20, input_dim=10))
model.add(Dense(40, input_dim=20))
model.add(Dense(60, input_dim=40))
model.add(Dense(80, input_dim=60))
model.add(Dense(100, input_dim=80))
model.add(Dense(120, input_dim=100))
model.add(Dense(140, input_dim=120))
model.add(Dense(99, input_dim=140))
model.add(Dense(77, input_dim=99))
model.add(Dense(50, input_dim=77))
model.add(Dense(30, input_dim=50))
model.add(Dense(12, input_dim=30))
model.add(Dense(1, input_dim=12))

# epochs =  100
# 로스 :  0.0001827188243623823
# 6의 예측값 :  [[5.969383]]

#3. 컴파일, 훈련
epochs = 100
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("=============================")
print("epochs = ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 : ", result)

# 로스 :  0.3805999755859375
# 로스 :  0.3812914788722992
# 6의 예측값 :  [[5.6273894]]
# 로스 :  0.38028597831726074
# 6의 예측값 :  [[5.728655]]
# 로스 :  0.394537299871444
# 6의 예측값 :  [[5.903086]]
