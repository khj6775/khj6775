# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#1. 데이터
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)

start1 = time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    height_shift_range=0.2,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    zoom_range=1.2,              # 축소 또는 확대
    shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

test_datagen = ImageDataGenerator(
    rescale=1./255,              # test 데이터는 수치화만!! 
)

path_train = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/"
path_test = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/"

xy_train = train_datagen.flow_from_directory(
    path_train,            
    target_size=(100,100),  
    batch_size=30000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100,100),
    batch_size=30000,            
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,  
)   


x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], 
                                                    test_size=0.2, 
                                                    random_state=528)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

print(x_train.shape, y_train.shape) # (20000, 100, 100, 3) (20000,)
print(x_test.shape, y_test.shape)   # (5000, 100, 100, 3) (5000,)
# 데이터 걸린시간 : 48.61 초

xy_test = xy_test[0][0]
# print(xy_test)
# print(xy_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(100, 100, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(filters=128,strides=1,padding='same', kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(units=8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras42/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k42_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.2,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)


### csv 파일 만들기 ###
y_submit = model.predict(xy_test)
# print(y_submit)

# y_submit = np.round(y_submit,4)
# print(y_submit)

print(y_submit)
sampleSubmission_csv['label'] = y_submit
sampleSubmission_csv.to_csv(path1 + "sampleSubmission_0803_1929.csv")


