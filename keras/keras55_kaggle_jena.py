import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU



#1. 데이터
Data = 'C:/AI5/_data/kaggle/jena/jena_climate_2009_2016.csv'

a = pd.read_csv(Data , index_col=0)

size = 720

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)      # append = 리스트 뒤에 붙인다.
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   

x = bbb[:, :-1]
y = bbb[:, -1,0]
print(x,y)
print(x.shape, y.shape)


