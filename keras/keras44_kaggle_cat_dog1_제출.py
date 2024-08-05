# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#1. 데이터
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


path_train = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/"
path_test = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/"



np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

print(x_train)
print(x_train.shape)    # (25000, 100, 100, 3)
print(y_train)  
print(y_train.shape)
print(x_test) 
print(x_test.shape) 
print(y_test) 
print(y_test.shape) 


# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=5289)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')
# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(64, (2,2), 
#                  activation='relu', 
#                  strides=1,padding='same',
#                  input_shape=(100, 100, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(2,2)))
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))         # 필터로 증폭, 커널 사이즈로 자른다.                              
# # model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
# # # model.add(BatchNormalization())
# # model.add(Dropout(0.25))
# model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

# model.add(Flatten())

# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(units=16, activation='relu'))

#                         # shape = (batch_size, input_dim)
# model.add(Dense(1, activation='sigmoid'))


# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                    patience=20, verbose=1,
#                    restore_best_weights=True,
#                    )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# path = './_save/keras42/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# filepath = "".join([path, 'k42_', date, '_', filename])   
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
#           validation_split=0.2,
#           callbacks=[es, mcp],
#           )
# end = time.time()

model = load_model('C:/AI5/_save/keras42/k42_0805_0030_0043-0.6083.hdf5')




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


### csv 파일 만들기 ###
y_submit = model.predict(x_test)
# print(y_submit)

# y_submit = np.round(y_submit,4)
# print(y_submit)

print(y_submit)
sampleSubmission_csv['label'] = y_submit
sampleSubmission_csv.to_csv(path1 + "teacher0805.csv")





