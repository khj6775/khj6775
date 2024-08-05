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




# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], test_size=0.2, random_state=5289)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')

# print(x_train.shape, y_train.shape) # (20000, 100, 100, 3) (20000,)
# print(x_test.shape, y_test.shape)   # (5000, 100, 100, 3) (5000,)

# xy_test = xy_test[0][0]
# # print(xy_test)
# # print(xy_test.shape)

