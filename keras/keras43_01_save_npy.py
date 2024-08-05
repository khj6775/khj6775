# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, Model
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
    batch_size=25000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)

xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100,100),
    batch_size=12500,            
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,  
)   


np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])



x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], test_size=0.2, random_state=5289)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

print(x_train.shape, y_train.shape) # (20000, 100, 100, 3) (20000,)
print(x_test.shape, y_test.shape)   # (5000, 100, 100, 3) (5000,)

xy_test = xy_test[0][0]
# print(xy_test)
# print(xy_test.shape)

