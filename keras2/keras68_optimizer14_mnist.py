from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

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

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

### 원핫1
from tensorflow.keras.utils import to_categorical

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
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(Dropout(0.1))

model.summary()

         # 필터로 증폭, 커널 사이즈로 자른다.                              
                                # shape = (batch_size, height, width, channels), (batch_size, rows, columns, channels)   
                                # shape = (batch_size, new_height, new_width, filters)
                                # batch_size 나누어서 훈련한다
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu', input_shape=(32,)))
                        # shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics = ['acc'])

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=5, verbose=0,
          batch_size=512,
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss[0]))

    y_predict = model.predict(x_test, verbose=0)
    acc = loss[1]
    print('lr : {0}, acc : {1}'.format(learning_rate, acc))

# =================1. 기본출력 ========================
# lr : 0.1, 로스 :2.3041820526123047
# lr : 0.1, acc : 0.11349999904632568
# =================1. 기본출력 ========================
# lr : 0.01, 로스 :2.3011481761932373
# lr : 0.01, acc : 0.11349999904632568
# =================1. 기본출력 ========================
# lr : 0.005, 로스 :2.301015853881836
# lr : 0.005, acc : 0.11349999904632568
# =================1. 기본출력 ========================
# lr : 0.001, 로스 :2.3009979724884033
# lr : 0.001, acc : 0.11349999904632568
# =================1. 기본출력 ========================
# lr : 0.0005, 로스 :2.3010008335113525
# lr : 0.0005, acc : 0.11349999904632568
# =================1. 기본출력 ========================
# lr : 0.0001, 로스 :2.3010027408599854
# lr : 0.0001, acc : 0.11349999904632568