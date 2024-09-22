from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이타
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
epochs = 2060
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
# 로스 :  0.3802859783172607
# 6의 예측값 :  [[5.728655]]
# 로스 :  0.3945372998714447
# 6의 예측값 :  [[5.903086]]
