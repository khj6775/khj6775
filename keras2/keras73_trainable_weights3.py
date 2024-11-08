#### 훈련후에 가중치로 1을 만들어 보자 ####

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.10.1

#1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

####################################################################
# model.trainable = False     # 동결 ★★★★★     역전파에서 갱신하지 않겠다. 
model.trainable = True     # 안동결 ★★★★★    디폴트 
####################################################################
print("============================================================================================")
print(model.weights)              
print("============================================================================================")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

print("============================================================================================")
print(model.weights)              
print("============================================================================================")
##### 위에 weghts로 손계산해서 1을 맹그러 #####
