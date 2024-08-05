# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset


import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
# np.save(np_path + 'keras45_01_brain_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras45_01_brain_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras45_01_brain_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras45_01_brain_y_test.npy', arr=xy_test[0][1])


x_train = np.load(np_path + 'keras45_07_gender_x_train.npy')
y_train = np.load(np_path + 'keras45_07_gender_y_train.npy')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=5289)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(100, 100, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(MaxPooling2D())
model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
model.add(Dropout(0.1))


# model.summary()

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


# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras45/07_save_npy_gender/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k45_gender_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

# start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

# end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
# print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)

# loss : 0.21999025344848633
# acc : 0.90909
# accuracy_score : 0.9090909090909091