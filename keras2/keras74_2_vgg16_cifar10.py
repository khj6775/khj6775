import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

#. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

## 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))

from tensorflow.keras.utils import to_categorical

### 원핫1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape)     
print(x_test.shape, y_test.shape)       

vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(32, 32, 3))    

vgg16.trainable=False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()

####### [실습] ########
# 비교할것
# 1. 이전에 본인이 한 최상의 결과와
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=Ture,(디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, trainable=False 
#### 위에 2,3번할때는 time 체크할 것  

import time

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

import datetime
date = datetime.datetime.now()       # 현재시간 저장

date = date.strftime("%m%d_%H%M")

start = time.time()
hist = model.fit(x_train, y_train, epochs=62, batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[]
          )

end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

from sklearn.metrics import r2_score, accuracy_score

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')


# 내가 한거
# loss : 1.0738571882247925
# acc : 0.64
# accuracy_score : 0.6359
# 걸린 시간 : 395.4 초

# trainable = True
# loss : 1.4340580701828003
# acc : 0.77
# accuracy_score : 0.7723
# 걸린 시간 : 278.97 초

# trainable = False
# loss : 1.2051922082901
# acc : 0.59
# accuracy_score : 0.5889
# 걸린 시간 : 105.17 초