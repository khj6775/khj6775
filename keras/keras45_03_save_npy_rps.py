
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = './_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=160,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

# batch_size=160
x_train = xy_train[0][0]
y_train = xy_train[0][1]

np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
np.save(np_path + 'keras45_03_rps_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_03_rps_y_train.npy', arr=xy_train[0][1])


#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(200, 200, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
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
model.add(Dense(3, activation='softmax'))


# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras45_03_rps/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k45_rps_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

# start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

end = time.time()
