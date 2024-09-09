# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# y는 (degC)로 잡아라
# 자르는 거 맘대로, 조건)pre = 2016.12.31 00:10부터 1.1까지 예측
# 144개.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder
import time

import os

start = time.time()

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# 데이터
data = pd.read_csv("C:\\ai5\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")
data = data.drop(["Date Time"], axis=1)
print(data.shape) #(420551, 15)

x = data.head(420407)
y_pre = data.tail(144)["T (degC)"]

y = x['T (degC)']
x = x.drop(["T (degC)"], axis =1)


size = 144
def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x = split_x(x, size)
y = split_x(y, size)

x_test1 = x[-1].reshape(-1,144,13)
# print(x)
x = np.delete(x, -1, axis =0)
y = np.delete(y, 0, axis = 0)

# y_pre = split_x(y ,size)

print(x.shape, y.shape) 
print(x_test1.shape)
# # print(y_pre)

# # x = np.delete(x, 1, axis=1)
# # y = x[1]
# # print(x.shape, y.shape) #(420264, 143, 13) (143, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

# #2. 모델 구성
# model = Sequential()
# model.add(LSTM(576, input_shape = (x.shape[1], x.shape[2])))
# model.add(Dense(576, activation='relu'))
# model.add(Dense(576, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(144))

# #3. 컴파일 및 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# model.compile(loss = 'mse', optimizer='adam')

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience=30,
#     restore_best_weights=True
# )

# ###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

path = 'C:/AI5/_save/keras55/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# filepath = "".join([path, 'k55_yena_', date, '_', filename])   
# #####################################

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode = 'auto',
#     verbose=1,
#     save_best_only=True,
#     filepath = filepath
# )

# model.fit(x_train, y_train,
#           epochs=1000,
#           batch_size=1024,
#           validation_split=0.2,
#           callbacks=[es,mcp])

model = load_model('C:/AI5/_save/keras55/k55_yena_0813_1150_0032-0.27930507.hdf5')

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test1)

print(loss, result)
print(result.shape)

from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, result):
    return np.sqrt(mean_squared_error(y_test, result))
 #  y_test, y_predict 매개변수
rmse = RMSE(y_pre.values.reshape(1, 144), result)

end = time.time()

print("RMSE : ", rmse)
print("걸린시간 : ", end-start, '초')


submit = pd.read_csv("C:\\AI5\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")

submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = result.reshape(144,1)
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

submit.to_csv(path + "jena_김호정_0813_1157.csv", index=False)

