import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, MaxPool2D
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#1. 데이터

# 이미지 폴더 수치화(2만개)
np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
# np.save(np_path + 'keras45_01_brain_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras45_01_brain_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras45_01_brain_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras45_01_brain_y_test.npy', arr=xy_test[0][1])

x_train1 = np.load(np_path + 'keras49_01_cat_and_dog_x_train.npy')
y_train1 = np.load(np_path + 'keras49_01_cat_and_dog_y_train.npy')
x_test1 = np.load(np_path + 'keras49_01_cat_and_dog_x_test.npy')
y_test1 = np.load(np_path + 'keras49_01_cat_and_dog_y_test.npy')


# 캐글 폴더 수치화(2.5만개)
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)
# # ## test 이미지 파일명 변경 ##
import os
import natsort

file_path = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test"
file_names = natsort.natsorted(os.listdir(file_path))

print(np.unique(file_names))
i = 1
for name in file_names:
     src = os.path.join(file_path,name)
     dst = str(i).zfill(5)+ '.jpg'
     dst = os.path.join(file_path, dst)
     os.rename(src, dst)
     i += 1


start1 = time.time()


np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

x = np.concatenate((x_train, x_train1))  # axis=0 default
y = np.concatenate((y_train, y_train1))  # axis=0 default 



print(x.shape, y.shape)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip=True,         # 수직 뒤집기 = 증폭
    width_shift_range=0.1,      # 평행 이동   = 증폭
    height_shift_range=0.1,     # 평행 이동 수직
    rotation_range=15,           # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,             # 축소 또는 확대
    shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode='nearest',        # 원래 있던 가까운 놈으로 채운다.
)

augment_size = 5000

print(x_train.shape[0]) 
randidx = np.random.randint(x.shape[0], size=augment_size)  # 60000, size=40000
print(randidx)  # [31344  4982 40959 ... 30622 14619 15678]
print(np.min(randidx), np.max(randidx))    # 1 27998

print(x_train[0].shape)  

x_augmented = x[randidx].copy()   # 카피하면 메모리 안전빵
y_augmented = y[randidx].copy()
print(x_augmented.shape, y_augmented.shape)    

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

# x_train = x_train.reshape(-1, 100, 100, 3)
# x_test = x_test.reshape(-1, 100, 100, 3)

print(x_train.shape, x_test.shape) 

x_train = np.concatenate((x_augmented, x), axis=0)  # axis=0 default
y_train = np.concatenate((y_augmented, y), axis=0)  # axis=0 default 

print(x_train.shape, y_train.shape)    

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    random_state=3752,
)


# x_test=x_test[0][0]


# #2. modeling
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
                                          
                        
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True)

################## mcp 세이브 파일명 만들기 시작 ###################
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\ai5\\_save\\keras49\\k49_05\\'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k49_05_', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)


start_time=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
# model.fit_generator(x_train, y_train,
#                     epochs=1000,
#                     verbose=1,
#                     callbacks=[es, mcp],
#                     validation_steps=50)
end_time=time.time()

# model.save('./_save/keras39/k39_07/keras39_07_mcp.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = np.round(model.predict(x_test, batch_size=16))
print("걸린 시간 :", round(end_time-start_time,2),'초')

# y_pre = np.round(y_pre)

### csv 파일 만들기 ###
y_submit = model.predict(x_test, batch_size=16)
# y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))
# print(y_submit)

# y_submit = np.round(y_submit)
# print(y_submit)

# print(y_submit)
sampleSubmission_csv['label'] = y_submit
sampleSubmission_csv.to_csv(path1 + "sample_Submission_0806_1803.csv")
