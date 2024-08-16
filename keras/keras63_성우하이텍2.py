import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, Conv1D, Flatten, Conv2D
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
PREDICT_DAYS = 30
FEATURE = 15
# -------------------------------------------------------------------------------------------------
x1_datasets = pd.read_csv(PATH_CSV + "NAVER 240816.csv", index_col = 0, thousands = ",") # 네이버

x1_datasets = x1_datasets[:CUT_DAYS].copy()

x1_datasets = x1_datasets.drop(['전일비'], axis = 1)

x1_datasets = x1_datasets.values.astype(np.float)

x1_datasets = x1_datasets[::-1]
# -------------------------------------------------------------------------------------------------
x2_datasets = pd.read_csv(PATH_CSV + "하이브 240816.csv", index_col = 0, thousands = ",") # 하이브

x2_datasets = x2_datasets.drop(['전일비'], axis = 1)

x2_datasets = x2_datasets.values.astype(np.float)

x2_datasets = x2_datasets[::-1]
# -------------------------------------------------------------------------------------------------
x3_datasets = pd.read_csv(PATH_CSV + "성우하이텍 240816.csv", index_col = 0, thousands = ",") # 성우하이텍

x3_datasets = x3_datasets[:CUT_DAYS].copy()

x3_datasets = x3_datasets.drop(['전일비'], axis = 1)

x3_datasets = x3_datasets.values.astype(np.float)

x3_datasets = x3_datasets[::-1]

y = np.array(x3_datasets[:, 3]) # 성우하이텍 종가
# -------------------------------------------------------------------------------------------------
x1_datasets = split_dataset(x1_datasets, PREDICT_DAYS)
x2_datasets = split_dataset(x2_datasets, PREDICT_DAYS)

x1_datasets = x1_datasets[:-1]
x2_datasets = x2_datasets[:-1]

y = split_dataset(y, 1)

y = y[PREDICT_DAYS:]
# -------------------------------------------------------------------------------------------------
x1_pred = x1_datasets[-1:]
x2_pred = x1_datasets[-1:]

answer = y[-1]
# -------------------------------------------------------------------------------------------------
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets,
    x2_datasets,
    y,
    train_size = 0.9,
    random_state = 9876
)

x1_train = x1_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
x2_train = x2_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)

#2-1 model
input01 = Input(shape=(PREDICT_DAYS, FEATURE, 1))

dense01 = Conv1D(10, 2, activation = 'relu', name = 'bit0101')(input01)
dense01 = Dense(20, activation = 'relu', name = 'bit0102')(dense01)
output01 = Dense(30, activation = 'relu', name = 'bit0103')(dense01)

#2-2 model
input02 = Input(shape=(PREDICT_DAYS, FEATURE, 1))

dense02 = Conv1D(10, 2, activation = 'relu', name = 'bit0201')(input02)
dense02 = Dense(20, activation = 'relu', name = 'bit0202')(dense02)
output02 = Dense(30, activation = 'relu', name = 'bit0203')(dense02)

merge01 = Concatenate(name = 'merge0101')([output01, output02])

merge01 = Flatten(name = 'merge0102')(merge01)
merge01 = Dense(10, activation = 'relu', name = 'merge0103')(merge01)
merge01 = Dense(20, activation = 'relu', name = 'merge0104')(merge01)
merge01 = Dense(30, activation = 'relu', name = 'merge0105')(merge01)

last_output = Dense(1, name = 'last')(merge01)

model = Model(inputs = [input01, input02], outputs = last_output)

model.summary()

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy', RootMeanSquaredError(name='rmse')])

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
    patience = 64,
    restore_best_weights = True
)

hist = model.fit(
    [x1_train, x2_train],
    y_train,
#    validation_split = 0.2,
    validation_data = ([x1_pred, x2_pred], answer),
    callbacks = [es, mcp],
    batch_size = 128,
    epochs = 10000
)

#4 predict
# model = load_model("./_save/k62_02.hdf5")

# x1_pre = np.array([range(3101, 3106), range(101, 106)]).T
#                        # 삼성 종가  하이닉스 종가
# x2_pre = np.array([range(3301, 3306), range(2501, 2506), range(4001, 4006)]).T
#                        # 원유, 환율, 금시세

eval = model.evaluate([x1_test, x2_test], [y_test, y_test])

print("loss :", eval)

result = model.predict([x1_pred, x2_pred])

print("예측값 :", result)