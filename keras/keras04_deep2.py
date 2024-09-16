from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss를 맹그러
# 에포는 100으로 고정, 건들지말것!!!
# 로스 기준 0.32 미만!

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
# model.add(Dense(12))
# model.add(Dense(1000))
# model.add(Dense(20))
# model.add(Dense(500))
# model.add(Dense(45))
# model.add(Dense(25))
# model.add(Dense(12))
# model.add(Dense(75))
# model.add(Dense(100))
model.add(Dense(3))
model.add(Dense(1, input_dim=3))

# epochs =  100
# 로스 :  0.3238484561443329
# 7의 예측값 :  [[6.806621]]

#3. 컴파일, 훈련
epochs = 100
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("=============================")
print("epochs = ", epochs)
print("로스 : ", loss)
result = model.predict([7])
print("7의 예측값 : ", result)

# 로스 :  0.3805999755859375
# 로스 :  0.3812914788722992
# 6의 예측값 :  [[5.6273894]]
# 로스 :  0.3802859783172607
# 6의 예측값 :  [[5.728655]]
# 로스 :  0.3945372998714447
# 6의 예측값 :  [[5.903086]]

# 로스 :  0.3239116370677948
# 6의 예측값 :  [[5.8744307]]