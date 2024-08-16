import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate, Conv1D, Flatten
from sklearn.model_selection import train_test_split

#1. 데이터
x1_datasets = pd.read_csv('C:\\ai5\\_data\\중간고사데이터\\NAVER 240816.csv', index_col=0, thousands = ',')
print(x1_datasets.shape)    # (5390, 17)
# data = data.drop(["Date Time"], axis=1)

x2_datasets = pd.read_csv('C:\\ai5\\_data\\중간고사데이터\\하이브 240816.csv', index_col=0, thousands = ',')
print(x2_datasets.shape)    # (948, 17)

x3_datasets = pd.read_csv('C:\\ai5\\_data\\중간고사데이터\\성우하이텍 240816.csv', index_col=0, thousands = ',')
print(x3_datasets.shape)    # (7058, 17)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x1_datasets['전일비'] = le.fit_transform(x1_datasets['전일비'])
x2_datasets['전일비'] = le.fit_transform(x2_datasets['전일비'])
x3_datasets['전일비'] = le.fit_transform(x3_datasets['전일비'])


# size = 5
def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x1_datasets = split_x(x1_datasets, 5)
x2_datasets = split_x(x2_datasets, 5)
x3_datasets = split_x(x3_datasets, 5)
y = split_x(x3_datasets, 5)



print(x1_datasets.shape)
print(x2_datasets.shape)
print(x3_datasets.shape)
print(y.shape)      # (7054, 5)
# exit()

x1_datasets = x1_datasets[:944]
x2_datasets = x2_datasets[:944]
x3_datasets = x3_datasets[:944]
y = y[:944]

# x_test1 = x3_datasets[-1].reshape(-1,5,16)  # 맨 마지막 x 로 평가  -1 = 맨마지막줄
# # print(x)
# x = np.delete(x3_datasets, -1, axis = 0)   # , 로 맨뒷줄 표현
# y = np.delete(y, 0, axis = 0)   # 0 = 첫번째줄

# x = x[ :143, : ]  # 인덱싱
# y = y[1:144, : ]


x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y,
                                                                         train_size=0.8, random_state= 215)

print(x1_train.shape, )
print(x2_train.shape, )
print(x3_train.shape,)
print(y_train.shape,)
# exit()

#2-1. 모델
input1 = Input(shape=(16,1))
dense1 = Conv1D(filters=10, kernel_size=2)(input1)
dense2 = (Conv1D(10, 2))(dense1)
dense3 = (Flatten())(dense2)

dense4 = Dense(64, activation='relu', name='bit2')(dense2)
dense5 = Dense(32, activation='relu', name='bit3')(dense4)
dense6 = Dense(16, activation='relu', name='bit4')(dense5)
output1 = Dense(8, activation='relu', name='bi5')(dense6)
model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

#2-2. 모델
input11 = Input(shape=(16,1))
dense11 = Conv1D(filters=10, kernel_size=2)(input11)
dense12 = (Conv1D(10, 2))(dense11)
# dense13 = (Flatten())(dense12)
dense14 = Dense(32, activation='relu', name='bit21')(dense12)
output11 = Dense(16, activation='relu', name='bit31')(dense14)
model2 = Model(inputs=input11, outputs=output11)

#2-3. 모델
input2 = Input(shape=(16,1))
dense21 = Conv1D(filters=10, kernel_size=2 )(input2)
dense22 = (Conv1D(10, 2))(dense21)
dense32 = (Flatten())(dense22)
dense33 = Dense(32, activation='relu', name='bit210')(dense32)
output111 = Dense(16, activation='relu', name='bit310')(dense33)
model3 = Model(inputs=input2, outputs=output111)



#2-3. 합체!!!
# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11, output111])
merge1 = Flatten()(merge1)
merge1 = Dense(8, name='mg2')(merge1)
merge1 = Dense(4, name='mg3')(merge1)
last_output = Dense(1, name='last')(merge1)


model = Model(inputs=[input1, input11, input2], outputs=last_output)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

###### mcp 세이브 파일명 만들기 ######

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=500,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras63/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k63_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit([x1_train, x2_train, x3_train], y_train,
          epochs=3000,
          batch_size=128,
          validation_split=0.2,
          callbacks=[es,mcp])

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test, verbose=1)
print('loss : ', loss)

y_pre1 = np.array([range(100,105), range(301,306)]).T
y_pre2 = np.array([range(201,206), range(411,416), range(150,155)]).T
y_pre3 = np.array([range(90,95), range(110,115),range(120,125),range(130,135)]).T

y_pre = model.predict([y_pre1, y_pre2, y_pre3])

print("3101~3105 까지 나와라\n", y_pre)