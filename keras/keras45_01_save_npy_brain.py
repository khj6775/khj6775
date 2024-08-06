
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

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
)

np_path = 'c:/AI5/_data/_save_npy/'     # 수치들의 형태는 다 넘파이다.
np.save(np_path + 'keras45_01_brain_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_01_brain_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras45_01_brain_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras45_01_brain_y_test.npy', arr=xy_test[0][1])
