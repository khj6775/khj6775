import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      # (50000, 32, 32, 3) (50000, 1) 


## 스케일링
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))

### 원핫1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(32, 32, 3))    

vgg16.trainable=False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))

model.summary()

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
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


# path ='./_save/keras38_04/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
# filepath = "".join([path, 'k38_04', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# # 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
# ##################### MCP 세이브 파일명 만들기 끝 ###########################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose = 1,
#     save_best_only=True,
#     filepath = filepath
# )


start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=512,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )

end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pre)
print('accuracy_score :', acc)
print("걸린 시간 :", round(end-start,2),'초')


# val_accuracy, max 로 바꿈, BatchNormalization 추가
# loss : 2.6442418098449707
# acc : 0.53
# accuracy_score : 0.5252
# 걸린 시간 : 2130.92 초

# False
# loss : 2.6408584117889404
# acc : 0.34
# accuracy_score : 0.3443
# 걸린 시간 : 68.54 초

# True
# loss : 2.615374803543091
# acc : 0.36
# accuracy_score : 0.3557
# 걸린 시간 : 124.78 초