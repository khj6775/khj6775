# 35_cnn2 copy

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, MaxPooling2D
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ####### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train))   #1.0 0.0

# ####### 스케일링 1-2
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5

# print(np.max(x_train), np.min(x_train))   # 1.0 -1.0

# ####### 스케일링 2. MinMaxScaler(), StandardScaler()
# x_train = x_train.reshape(60000, 28*28)   # 2차원으로 먼저 Reshape
# x_test = x_test.reshape(10000, 28*28)  

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))  

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

### 원핫1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

# ### 원핫2
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# ### 원핫3
# from sklearn.preprocessing import OneHotEncoder     # 전처리
# ohe = OneHotEncoder(sparse=False)    # True가 디폴트 
# y_train = y_train.reshape(-1, 1)
# y_trian = ohe.fit_transform(y_train)   # -1 은 데이터 수치의 끝 
# y_test = y_test.reshape(-1, 1)
# y_test = ohe.fit_transform(y_test)   # -1 은 데이터 수치의 끝 


np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)   흑백데이터라 맨뒤에 1이 생략 -- 변환시켜준다
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)


# print(y_.shape)  

# print(pd.value_counts(y))

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=8888)



#2. 모델
model = Sequential()
model.add(Dense(280, input_shape=(28,28)))           # (n, 28, 28)
model.add(Reshape(target_shape=(28, 28, 10) ))       # (n, 28, 28, 1)
model.add(Conv2D(64, (3,3)))    # 26, 26, 64 
model.add(MaxPooling2D())       # 13, 13, 64
model.add(Conv2D(5, (4,4), ))   # 10, 10, 5

# model.add(Reshape(target_shape=(10 * 10, 5)))
model.add(Reshape(target_shape=(10 * 10 * 5, )))    # (500, )



# model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu', input_shape=(32,)))
                        # shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()       # 현재시간 저장
print(date)         # 2024-07-26 16:50:37.570567
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date)) # <class 'str'>


path ='./_save/keras35_04/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 1000-0.7777.hdf5    { } = dictionary, 키와 밸류  d는 정수, f는 소수
filepath = "".join([path, 'k35_04', date, '_', filename])      # 파일위치와 이름을 에포와 발로스로 나타내준다
# 생성 예 : ""./_save/keras29_mcp/k29_1000-0.7777.hdf5"
##################### MCP 세이브 파일명 만들기 끝 ###########################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)


start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )

end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

# loss : 0.053418297320604324
# acc : 0.99
# accuracy_score : 0.9867
# 걸린 시간 : 236.21 초
