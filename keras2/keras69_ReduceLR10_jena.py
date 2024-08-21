import tensorflow as tf
import random as rn
import numpy as np
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Bidirectional, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
path1 = "C:/ai5/_data/kaggle/jena/"
datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)

print(datasets.shape)   # (420551, 14)

y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)

## 훈련할 데이터 자르기
x_data = datasets[:-288].drop(['T (degC)'], axis=1)
y_data = datasets[144:-144]['T (degC)']

print(x_data)       # [420407 rows x 13 columns]
print(y_data)       # Name: T (degC), Length: 420407, dtype: float64

size_x = 144 
size_y = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

# x = x[:-1]
# y = y[(size_x-size_y+1):]


print(x.shape, y.shape)     # (420120, 144, 13) (420120, 144)
# print(x, y)

# 예측을 위한 x 데이터 
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
x_predict = x_predict.to_numpy()
print(x_predict)
print(x_predict.shape)  # (144, 13)
# x_predict = split_x(x_predict, size_x)
print(x_predict.shape)  # (144, 13)

x_predict = x_predict.reshape(1,144,13)

# print(y[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2532)


print(x_train.shape)        # (378108, 144, 13)
print(x_test.shape)         # (42012, 144, 13)

# ## 스케일링 추가 ###
from sklearn.preprocessing import StandardScaler
x_train = x_train.reshape(378108,144*13)
x_test = x_test.reshape(42012,144*13)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_predict = x_predict.reshape(144,13)
# x_predict = scaler.transform(x_predict)
# x_predict = x_predict.reshape(1,144*13)

from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[1])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 69
# 0.99 이상 : 155
# 0.999 이상 : 227
# 1.0 일 때 : 1872


num = [69,155,227,1872]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델 구성
    model = Sequential()
    model.add(Dense(1024, input_shape=(num[i],)))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(144))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                        patience=25, verbose=1,
                        factor=0.8,)   # factor = lr 줄여주는 비율


from tensorflow.keras.optimizers import Adam

learning_rate = 0.001      # default = 0.001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),)

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=10,
          batch_size=512,
          callbacks=[es,rlr],
          )

#4. 평가,예측
print("=================1. 기본출력 ========================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

# =================1. 기본출력 ========================
# lr : 0.001, 로스 :3.706329822540283
# lr : 0.001, acc : 0.9475359825391604