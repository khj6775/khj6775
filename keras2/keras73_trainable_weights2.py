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
model.trainable = False     # 동결 ★★★★★     역전파에서 갱신하지 않겠다. 
# model.trainable = True     # 안동결 ★★★★★    디폴트 
####################################################################
print("============================================================================================")
print(model.weights)              
print("============================================================================================")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=3, epochs=1000, verbose=0)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

# True
# [[1.0000002]       
#  [2.       ]
#  [2.9999995]
#  [4.       ]
#  [5.       ]]

# False
# [[0.45656443]
#  [0.91312885]
#  [1.369693  ]
#  [1.8262577 ]
#  [2.2828221 ]]