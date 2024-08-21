
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = './_data/image/cat_and_dog/train/'
path_test = './_data/image/cat_and_dog/test/'


xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(50, 50),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(50,50),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
)

x_train, x_test, y_train, y_test = train_test_split(
            xy_train[0][0],
            xy_train[0][1],
            train_size=0.8,
            shuffle=False
)

end = time.time()


#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(50, 50, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.summary()

         # 필터로 증폭, 커널 사이즈로 자른다.                              
                                # shape = (batch_size, height, width, channels), (batch_size, rows, columns, channels)   
                                # shape = (batch_size, new_height, new_width, filters)
                                # batch_size 나누어서 훈련한다
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                        patience=25, verbose=1,
                        factor=0.8,)   # factor = lr 줄여주는 비율


from tensorflow.keras.optimizers import Adam

learning_rate = 0.001      # default = 0.001

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=10,
          batch_size=512,
          callbacks=[es,rlr],
          )

#4. 평가,예측
print("=================1. 기본출력 ========================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr : {0}, 로스 :{1}'.format(learning_rate, loss[0]))

y_predict = model.predict(x_test, verbose=0)
acc = loss[1]
print('lr : {0}, acc : {1}'.format(learning_rate, acc))