import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


path = 'C:/AI5/_data/kaggle/jena/jena_climate_2009_2016.csv'

x = pd.read_csv(path, index_col=0)

print(x.shape) #(420551, 14)

x = x[:-144]
y = x['T (degC)']
x = x.drop(['T (degC)'], axis=1)

 
print(x.shape)  #(420407, 13)
print(y.shape) #(420407,)

size = 1440
size2 = 144

def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array (aaa)

def split_x2(dataset, size2):
    aaa= []
    for i in range(len(dataset) - size2 + 1):
        subset = dataset[i : (i + size2)]
        aaa.append(subset)
    return np.array (aaa)

xxx = split_x(x, size)

print(xxx.shape) #(420264, 144, 13)

yyy = split_x(y, size2)
# print(yyy)
print(yyy.shape) #(420264, 144)

yyy = yyy[1:]
print(yyy.shape) #(420263, 144)
# print(yyy) 

x_test = xxx[-1]
# print(x_test)
print(x_test.shape) #(144, 13)

xxx = xxx[:-1]
print(xxx.shape) #(420263, 144, 13)


model = Sequential()
model.add(LSTM(32, return_sequences= True, input_shape=(144, 13)))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(xxx, yyy, epochs=1, batch_size=1, verbose=1, ) 

loss = model.evaluate(xxx, yyy)
x_test = x_test.reshape(1, 144, 13)
y_predict = model.predict(x_test)

print('로스 : ', loss)
print('y_predict', y_predict)
