# 35_cnn2 copy

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

#. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

np.set_printoptions(edgeitems=30, linewidth = 1024)

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)   흑백데이터라 맨뒤에 1이 생략 -- 변환시켜준다
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# print(y_.shape)  

# print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=8888)



#2. 모델
model = Sequential()
model.add(Conv2D(10, (3,3), input_shape=(28, 28, 1),
                 strides=1,
                #  padding='same'
                #  padding='valid'  #default
                 ))     # 26, 26, 10

model.add(MaxPooling2D())   # 13, 13, 10
model.add(Conv2D(filters=9, kernel_size=(3,3), #11, 11, 9
                 strides=1,
                 padding='valid'
                 ))                               
model.add(Conv2D(8, (2,2)))                              
# model.add(Flatten())

# model.add(Dense(units=8))
# model.add(Dense(units=9, input_shape=(8,)))
#                         # shape = (batch_size, input_dim)
# model.add(Dense(10, activation='softmax'))

model.summary()
'''
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=2048,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
'''