
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM, Reshape,Flatten,Conv1D
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
# np.save(np_path + 'keras45_01_brain_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras45_01_brain_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras45_01_brain_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras45_01_brain_y_test.npy', arr=xy_test[0][1])


x_train = np.load(np_path + 'keras45_03_rps_x_train.npy')
y_train = np.load(np_path + 'keras45_03_rps_y_train.npy')

train_datagen =  ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=1.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

augment_size = 10000

print(x_train.shape[0]) # 60000
randidx = np.random.randint(x_train.shape[0], size=augment_size)  # 60000, size=40000
print(randidx)  # [31344  4982 40959 ... 30622 14619 15678]
print(np.min(randidx), np.max(randidx))    # 0 59995

print(x_train[0].shape)     # (28,28)

x_augmented = x_train[randidx].copy()     # 카피하면 메모리 안전빵
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape)     # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],   # 40000
    x_augmented.shape[1],   # 28
    x_augmented.shape[2], 3)  # 28
print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)    

# x_train = x_train.reshape(-1, 100, 100, 3)
# x_test = x_test.reshape(-1, 100, 100, 3)

print(x_train.shape)  

x_train = np.concatenate((x_augmented, x_train), axis=0)  # axis=0 default
y_train = np.concatenate((y_augmented, y_train), axis=0)  # axis=0 default 

print(x_train.shape, y_train.shape)    

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=5289)


#2. 모델
model = Sequential()

# model.add(Conv1D(64,2))
model.add(Conv1D(filters=64, kernel_size=2, input_shape=(100,100, 3)))
model.add(Conv1D(32, 2))

# model.add(Conv2D(64, (3,3), 
#                  activation='relu', 
#                  strides=1,padding='same',
#                  input_shape=(100, 100, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
# model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
# model.add(Dropout(0.1))

# model.add(Reshape(target_shape=(25*25, 32) ))

# model.add(LSTM(units=32, input_shape=(25*25,32),return_sequences=True))
# model.add(LSTM(32))
        
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(3, activation='softmax'))


# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
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

path = './_save/keras49/08_save_rps/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k49_rps_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

# start = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

end = time.time()

# model = load_model('C:/AI5/_save/keras45_03_rps/k45_rps_0805_1252_0019-0.4363.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test, batch_size=16)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
# print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)


# LSTM
# loss : 1.0976650714874268
# acc : 0.33349

# Conv1D
# loss : 1.0971804857254028171
# acc : 0.35563