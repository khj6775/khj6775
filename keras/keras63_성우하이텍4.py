import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, Conv1D, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError

PATH_CSV = "C:/ai5/_data/중간고사데이터/"

#1 data
#=================================================================
def split_dataset(dataset, size):
    result = []

    for i in range(len(dataset) - size + 1):
        if (i % 10000 == 0):
            print(i)

        subset = dataset[i : (i + size)]

        result.append(subset)

    return np.array(result)
#=================================================================

CUT_DAYS = 948
PREDICT_DAYS = 2
FEATURE = 12
# -------------------------------------------------------------------------------------------------
x1_datasets = pd.read_csv(PATH_CSV + "NAVER 240816.csv", index_col = 0, thousands = ",") # 네이버

x1_datasets = x1_datasets.drop(['전일비', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis = 1)

x1_datasets = x1_datasets[:CUT_DAYS].copy()

x1_dt = pd.to_datetime(x1_datasets.index, format = '%Y/%m/%d')

x1_datasets['day'] = x1_dt.day
x1_datasets['month'] = x1_dt.month
x1_datasets['year'] = x1_dt.year
x1_datasets['dow'] = x1_dt.dayofweek

x1_datasets = x1_datasets.values.astype(np.float)

x1_datasets[0, 0] = 159200.0
x1_datasets[0, 1] = 159200.0
x1_datasets[0, 2] = 157000.0
x1_datasets[0, 3] = 157500.0

x1_datasets[0, 4] = 200.0
x1_datasets[0, 5] = 0.13
x1_datasets[0, 6] = 813296.0
x1_datasets[0, 7] = 128666.0

x1_datasets = x1_datasets[::-1]
# -------------------------------------------------------------------------------------------------
x2_datasets = pd.read_csv(PATH_CSV + "하이브 240816.csv", index_col = 0, thousands = ",") # 하이브

x2_datasets = x2_datasets.drop(['전일비', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis = 1)

x2_dt = pd.to_datetime(x2_datasets.index, format = '%Y/%m/%d')

x2_datasets['day'] = x2_dt.day
x2_datasets['month'] = x2_dt.month
x2_datasets['year'] = x2_dt.year
x2_datasets['dow'] = x2_dt.dayofweek

x2_datasets = x2_datasets.values.astype(np.float)

x2_datasets[0, 0] = 164700.0
x2_datasets[0, 1] = 168600.0
x2_datasets[0, 2] = 163500.0
x2_datasets[0, 3] = 166400.0

x2_datasets[0, 4] = 3300.0
x2_datasets[0, 5] = 2.02
x2_datasets[0, 6] = 188123.0
x2_datasets[0, 7] = 31347.0

x2_datasets = x2_datasets[::-1]
# -------------------------------------------------------------------------------------------------
x3_datasets = pd.read_csv(PATH_CSV + "성우하이텍 240816.csv", index_col = 0, thousands = ",") # 성우하이텍

x3_datasets = x3_datasets.drop(['전일비', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis = 1)

x3_datasets = x3_datasets[:CUT_DAYS].copy()

x3_dt = pd.to_datetime(x3_datasets.index, format = '%Y/%m/%d')

x3_datasets['day'] = x3_dt.day
x3_datasets['month'] = x3_dt.month
x3_datasets['year'] = x3_dt.year
x3_datasets['dow'] = x3_dt.dayofweek

x3_datasets = x3_datasets.values.astype(np.float)

x3_datasets[0, 0] = 7580.0
x3_datasets[0, 1] = 7630.0
x3_datasets[0, 2] = 7350.0
x3_datasets[0, 3] = 7420.0

x3_datasets[0, 4] = 30.0
x3_datasets[0, 5] = 0.41
x3_datasets[0, 6] = 833336.0
x3_datasets[0, 7] = 6207.0

x3_datasets = x3_datasets[::-1]
# -------------------------------------------------------------------------------------------------
y = np.array(x3_datasets[:, 3]) # 성우하이텍 종가
# -------------------------------------------------------------------------------------------------
x1_datasets = split_dataset(x1_datasets, PREDICT_DAYS)
x2_datasets = split_dataset(x2_datasets, PREDICT_DAYS)
x3_datasets = split_dataset(x3_datasets, PREDICT_DAYS)

x1_datasets = x1_datasets[:-1]
x2_datasets = x2_datasets[:-1]

x3_datasets = x3_datasets[:-1]

y = split_dataset(y, 1)

y = y[PREDICT_DAYS:]
# -------------------------------------------------------------------------------------------------
x1_pred = x1_datasets[-1:]
x2_pred = x2_datasets[-1:]
x3_pred = x3_datasets[-1:]

answer = np.array([7420,])
# -------------------------------------------------------------------------------------------------
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets,
    x2_datasets,
    x3_datasets,
    y,
    train_size = 0.9,
    random_state = 7777
)

x1_train = x1_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
x2_train = x2_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
x3_train = x3_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)

#2-1 model
input01 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

dense01 = Conv2D(10, 2, activation = 'relu', name = 'bit0101', padding = 'same')(input01)
dense01 = MaxPooling2D()(dense01)
dense01 = Dense(20, activation = 'relu', name = 'bit0102')(dense01)
output01 = Dense(30, activation = 'relu', name = 'bit0103')(dense01)

#2-2 model
input02 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

dense02 = Conv2D(10, 2, activation = 'relu', name = 'bit0201', padding = 'same')(input02)
dense02 = MaxPooling2D()(dense02)
dense02 = Dense(20, activation = 'relu', name = 'bit0202')(dense02)
output02 = Dense(30, activation = 'relu', name = 'bit0203')(dense02)

#2-3 model
input03 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

dense03 = Conv2D(10, 2, activation = 'relu', name = 'bit0301', padding = 'same')(input03)
dense03 = MaxPooling2D()(dense03)
dense03 = Dense(20, activation = 'relu', name = 'bit0302')(dense03)
output03 = Dense(30, activation = 'relu', name = 'bit0303')(dense03)
# -------------------------------------------------------------------------------------------------
merge01 = Concatenate(name = 'merge0101')([output01, output02, output03])

merge01 = Flatten(name = 'merge0102')(merge01)
merge01 = Dense(10, activation = 'relu', name = 'merge0103')(merge01)
merge01 = Dense(20, activation = 'relu', name = 'merge0104')(merge01)
merge01 = Dense(30, activation = 'relu', name = 'merge0105')(merge01)

last_output = Dense(1, name = 'last')(merge01)

model = Model(inputs = [input01, input02, input03], outputs = last_output)

model.summary()

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy', RootMeanSquaredError(name = 'rmse')])
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras63/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([PATH, 'k63_', date, "_", filename])
##################### mcp 세이브 파일명 만들기 끝 ###################
mcp = ModelCheckpoint(
    monitor = 'val_rmse',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'val_rmse',
    mode = 'min',
    patience = 16,
    restore_best_weights = True
)

class OnEpochEndPred(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        print("예측값 :", model.predict([x1_pred, x2_pred, x3_pred]))

hist = model.fit(
    [x1_train, x2_train, x3_train],
    y_train,
#    validation_split = 0.25,
    validation_data = ([x1_pred, x2_pred, x3_pred], answer),
    callbacks = [es, mcp, OnEpochEndPred()],
    batch_size = 128,
    epochs = 10000
)

#4 predict
# model = load_model("./_save/k63.hdf5")

eval = model.evaluate([x1_test, x2_test, x3_test], y_test)

print("loss :", eval)

result = model.predict([x1_pred, x2_pred, x3_pred])

print("7420 나와라 :", result)

# 예측값 : [[7439.796]]


# 예측값 : [[7419.532]]