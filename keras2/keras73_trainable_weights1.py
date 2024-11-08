import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.10.1

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
"""

print(model.weights)

"""
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.5984144,  0.0132283, -0.5460101]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy= 
array([[-0.5503688 , -0.02419055],  
       [-0.37951523,  0.21739638],
       [ 0.58901846, -0.4752571 ]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[ 0.9757787 ],
       [-0.61850667]], dtype=float32)>, 
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

히든레이어 층별 가중치 값.
kernel = 가중치

"""

print("=================================================================================================================================")
print(model.trainable_weights)
# 위와 똑같은 결과 출력

print("=================================================================================================================================")
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

####################################################################
model.trainable = False     # 동결 ★★★★★     역전파에서 갱신하지 않겠다. 
####################################################################
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

print("====================== model.weights =================================================================================================")
print(model.weights)              
print("============================== model.trainable_weights ===============================================================")
print(model.trainable_weights) 
print("======================================================================================")

model.summary()

# Total params: 17
# Trainable params: 0       동결, freeze
# Non-trainable params: 17

