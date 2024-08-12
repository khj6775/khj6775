# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# y는 (degC)로 잡아라
# 자르는 거 맘대로, 조건)pre = 2016.12.31 00:10부터 1.1까지 예측
# 144개
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Flatten, BatchNormalization, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import time
import os
from sklearn.preprocessing import StandardScaler

start = time.time()

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# 데이터
data = pd.read_csv(".\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")
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

x_test1 = x[-1].reshape(-1,144,13)  # 맨 마지막 x 로 평가  -1 = 맨마지막줄
# print(x)
x = np.delete(x, -1, axis = 0)   # , 로 맨뒷줄 표현
y = np.delete(y, 0, axis = 0)   # 0 = 첫번째줄
# x = x[ :143, : ]  # 인덱싱
# y = y[1:144, : ]


# y_pre = split_x(y ,size)

print(x.shape, y.shape) 
print(x_test1.shape)
# # print(y_pre)

# # x = np.delete(x, 1, axis=1)
# # y = x[1]
# # print(x.shape, y.shape) #(420264, 143, 13) (143, 13)

# scaler = StandardScaler()
# x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
# x_test1 = scaler.transform(x_test1.reshape(-1, x_test1.shape[-1])).reshape(x_test1.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

# 2. 모델구성
model = Sequential()
model.add(Bidirectional(LSTM(144), input_shape = (x.shape[1], x.shape[2])))
# model.add(LSTM(144))
# Flatten()

model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))

model.add(Dense(144))


#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss = 'mse', optimizer='adam',)
            #   metrics = [tf.keras.metrics.RootMeanSquaredError(name='rmse')])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=40,
    restore_best_weights=True
)

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras55/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k55_yena_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x_train, y_train,
          epochs=3000,
          batch_size=1024,
          validation_split=0.2,
          callbacks=[es,mcp])

# model = load_model('C:\\ai5\\_save\\keras55\\jena_김호정.hdf5')

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test1)
result = np.array([result]).reshape(144,1)
# result = np.reshape(result, (144,1))

print(loss, result)
print(result.shape)

def RMSE(y_test, result):
    return np.sqrt(mean_squared_error(y_test, result))
 #  y_test, y_predict 매개변수
rmse = RMSE(y_pre, result)

end = time.time()

print("RMSE : ", rmse)
print("걸린시간 : ", end-start, '초')


submit = pd.read_csv(".\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")

submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = result.reshape(144,1)
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

submit.to_csv("C:\\ai5\\_save\\keras55\\jena_김호정.csv", index=False)


# RMSE :  1.2593521040329536
# 걸린시간 :  2202.020427942276 초

