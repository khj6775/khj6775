# 배치를 160으로 잡고
# x, y를 추출해서 모델을 맹그러봐
# acc 0.99 이상

import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM, Reshape, Conv1D, Flatten
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = 'C:/AI5/_data/image/horse_human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),
    batch_size=1050,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(
            xy_train[0][0],
            xy_train[0][1],
            train_size=0.8,
            shuffle=False
)

#2. 모델
model = Sequential()


model.add(Conv1D(filters=64, kernel_size=2, input_shape=(150,150, 3)))
model.add(Conv1D(32, 2))

# model.add(Conv2D(64, (3,3), 
#                  activation='relu', 
#                  strides=1,padding='same',
#                  input_shape=(150, 150, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
# model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
# model.add(Dropout(0.1))

# model.summary()

# model.add(Reshape(target_shape=(27*27, 128) ))

# model.add(LSTM(units=32, input_shape=(27*27,128),return_sequences=True))
# model.add(LSTM(32))
        
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(1, activation='sigmoid'))


model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=160,
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

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)


# loss : 0.000754596432670
# acc : 1.0
# acc_score : 1.0

# LSTM
# loss : 0.6879040598869324
# acc : 0.5316
# 걸린 시간 : 18.67 초

# Conv1D
# loss : 0.0505344197154045
# acc : 0.9854