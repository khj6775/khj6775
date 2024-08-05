# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset


import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip=True,         # 수직 뒤집기 = 증폭
    width_shift_range=0.1,      # 평행 이동   = 증폭
    height_shift_range=0.1,     # 평행 이동 수직
    rotation_range=5,           # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,             # 축소 또는 확대
    shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = 'c:/AI5/_data/kaggle/biggest gender/faces/'


xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=28000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

# print(xy_train.class_indices)
# exit()      # {'man': 0, 'woman': 1}

np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
np.save(np_path + 'keras45_07_gender_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_07_gender_y_train.npy', arr=xy_train[0][1])

