import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.
# 판다스 데이터 변경 숙달 시키기.

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip=True,         # 수직 뒤집기 = 증폭
    width_shift_range=0.1,      # 평행 이동   = 증폭
    height_shift_range=0.1,     # 평행 이동 수직
    rotation_range=15,           # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,             # 축소 또는 확대
    shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode='nearest',        # 원래 있던 가까운 놈으로 채운다.
)

augment_size = 40000

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
    x_augmented.shape[2], 1)  # 28
print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)     # (40000, 28, 28, 1) 변환된 데이터

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_augmented, x_train), axis=0)  # axis=0 default
y_train = np.concatenate((y_augmented, y_train), axis=0)  # axis=0 default 

print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)

### 원핫
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

np.set_printoptions(edgeitems=30, linewidth = 1024)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(28, 28, 1)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(MaxPooling2D())
model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.25))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(units=1026, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))

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

path = './_save/keras49/01_fashion/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k49_fashion_', date, '_', filename])   
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

end = time.time()

# model = load_model('C:/AI5/_save/keras45_03_rps/k45_rps_0805_1252_0019-0.4363.hdf5')


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

# loss : 0.2701088786125183
# acc : 0.907
# accuracy_score : 0.8961






'''
print(x_train.shape)    # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)

# plt.imshow(x_train[0], cmap='gray')     # cmap 색깔 변신
# plt.show()

aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape)

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False,
).next()

print(xy_data)
print(type(xy_data))  # <class 'tuple'>

# print(x_data.shape)     # AttributeError: 'tuple' object has no attribute 'shape', len으로 확인
print(len(xy_data))       # 2   튜플안에 두개의 넘파이가 들어가있다.(x,y)
print(xy_data[0].shape)   # (100, 28, 28, 1)
print(xy_data[1].shape)   # (100, )

plt.figure(figsize=(7,7))   # 7인치
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()
'''