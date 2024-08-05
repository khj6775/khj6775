import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온거 수치화
import matplotlib.pyplot as plt

me=np.load('C:\\AI5\\_data\\image\\me\\keras46_image_me.npy')

model = load_model('C:/AI5/_save/keras42/k42_0805_0030_0043-0.6083.hdf5')

y_pred = np.round(model.predict(me))

print(y_pred)



