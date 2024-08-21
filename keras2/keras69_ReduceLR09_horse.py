import tensorflow as tf
import random as rn
import numpy as np
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling1D, Bidirectional, LSTM, Conv1D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게


#1. 데이터
train_datagen =  ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

start1 = time.time()
np_path = "c:/ai5/_data/_save_npy/"

x_train = np.load(np_path + 'keras45_02_horse_x_train.npy')
y_train = np.load(np_path + 'keras45_02_horse_y_train.npy')


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=337)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

print(x_train.shape, y_train.shape) # (821, 200, 200, 3) (821,)
print(x_test.shape, y_test.shape)   # (206, 200, 200, 3) (206,)
# 데이터 걸린시간 : 71.87 초


augment_size = 10000  

randidx = np.random.randint(x_train.shape[0], size = augment_size) 
print(randidx)              
print(np.min(randidx), np.max(randidx)) 

print(x_train[0].shape) 

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   
print(y_augmented.shape)   

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],      
    x_augmented.shape[1],     
    x_augmented.shape[2], 3)    

print(x_augmented.shape)  

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)   

x_train = x_train.reshape(-1, 100, 100, 3)
x_test = x_test.reshape(-1, 100, 100, 3)

print(x_train.shape, x_test.shape) 

# ## numpy에서 데이터 합치기
# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (10821, 200, 200, 3) (10821,)

print(np.unique(y_train, return_counts=True))

x_train = x_train.reshape(-1,100*100*3)
x_test = x_test.reshape(-1,100*100*3)

from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[0])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#2. 모델 구성
model = Sequential()   
model.add(Dense(1024, input_shape=(440,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
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

# lr : 0.001, 로스 :0.9636828303337097
# lr : 0.001, acc : 0.581818163394928