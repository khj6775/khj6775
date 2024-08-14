import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T       # (100, 2)
                        # 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411,511),
                        range(150,250)]).transpose()         # (100, 3)
                        # 원유, 환율, 금시세

y = np.array(range(3001, 3101))  # 한강의 화씨 온도.

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y,
                                                                         train_size=0.8, random_state= 215)

print(x1_train.shape, x1_test)
print(x2_train.shape, x2_test)
print(y_train.shape, y_test)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(40, activation='relu', name='bit4')(dense3)
output1 = Dense(50, activation='relu', name='bi5')(dense4)
model1 = Model(inputs=input1, outputs=output1)
model1.summary()

#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense21 = Dense(200, activation='relu', name='bit21')(dense11)
output11 = Dense(300, activation='relu', name='bit31')(dense21)
model2 = Model(inputs=input11, outputs=output11)

#2-3. 합체!!!
# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11])
merge2 = Dense(8, name='mg2')(merge1)
merge3 = Dense(4, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11], outputs=last_output)
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

path = './_save/keras62/ensemble1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k62_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit([x1_train, x2_train,], y_train,
          epochs=3000,
          batch_size=128,
          validation_split=0.2,
          callbacks=[es,mcp])

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test, verbose=1)
print('loss : ', loss)

y_pre1 = np.array([range(100,105), range(301,306)]).T
y_pre2 = np.array([range(201,206), range(411,416), range(150,155)]).T

y_pre = model.predict([y_pre1, y_pre2])


# print("3101~3105 까지 나와라\n", y_pre)

# loss :  0.02799764834344387