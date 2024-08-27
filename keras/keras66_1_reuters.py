from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Embedding
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,
    # maxlen=10,
    test_split=0.2,
)
print(x_train)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(y_train)      # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train))
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(len(np.unique(y_train)))  # 46
# print(len(np.unique(x_train)))

print(type(x_train))    # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>
print(len(x_train[0]), len(x_train[1]))     # 87 56

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)  )  # 2376
print("뉴스기사의 최소길이 : ", min(len(i) for i in x_train)  )  # 13
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) /len(x_train)) # 145.5398574927633

# sum(map(len, x_train)) /len(x_train) 공부하고 이해하기

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating='pre')

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()

# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
                        
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)


#2. 모델 구성 
model = Sequential()
model.add(Embedding(1000,100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras66_1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k66_1', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=128,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es,mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# loss : 1.18719482421875
# acc : 0.7016919255256653
