import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x_datasets = np.array([range(100), range(301,401)]).T       # (100, 2)
                        # 삼성 종가, 하이닉스 종가

y1 = np.array(range(3001, 3101))  # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))  # 비트코인 가격.

x_train, x_test, y1_train, y1_test, \
    y2_train, y2_test = train_test_split(                                   
    x_datasets, y1, y2, train_size=0.8, random_state= 528)

print(x_train.shape, x_test)
print(y1_train.shape)
print(y2_train.shape)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(64, activation='relu', name='bit1')(input1)
dense2 = Dense(128, activation='relu', name='bit2')(dense1)
dense3 = Dense(64, activation='relu', name='bit3')(dense2)
dense4 = Dense(32, activation='relu', name='bit4')(dense3)
output1 = Dense(16, activation='relu', name='bi5')(dense4)
model1 = Model(inputs=input1, outputs=output1)
last_output1 = Dense(1, name='last1')(output1)   # y 값 두개로 나눈다
last_output2 = Dense(1, name='last2')(output1)
# model1.summary()


# #2-4. 합체!!!
# # merge1 = concatenate([output1, output11], name='mg1')
# merge1 = Concatenate(name='mg1')([output1,])
# merge2 = Dense(8, name='mg2')(merge1)
# merge3 = Dense(8, name='mg3')(merge2)
# last_output1 = Dense(1, name='last1')(merge3)   # y 값 두개로 나눈다
# last_output2 = Dense(1, name='last2')(merge3)

# middle_output = Dense(1, name='last')(merge3)

# #1 model = Model(inputs=[input1, input11, input2], outputs=[last_output1, last_output2])
# #2 model = Model(inputs=[input1, input11, input2], outputs=middle_output)
# # model.summary()

# # 2-5. 분기1.
# dense31 = Dense(8, activation='relu', name='bit41')(middle_output)
# dense32 = Dense(4, activation='relu', name='bit42')(dense31)
# last_output1 = Dense(1, name='bit43')(dense32)

# # 2-6. 분기2.
# dense33 = Dense(4, activation='relu', name='bit44')(middle_output)
# dense34 = Dense(2, activation='relu', name='bit45')(dense33)
# last_output2 = Dense(1, name='bit46')(dense34)

# 2-7. 다시 합체

model = Model(inputs=input1, outputs=[last_output1, last_output2])

#3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train], epochs=1000, batch_size=64)
model.compile(loss = 'mse', optimizer='adam',)
            #   metrics = [tf.keras.metrics.RootMeanSquaredError(name='rmse')])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=500,
    restore_best_weights=True
)

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras62/'
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

model.fit(x_train, [y1_train,y2_train],
          epochs=3000,
          batch_size=1024,
          validation_split=0.2,
          callbacks=[es,mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test], verbose=1)
print('loss : ', loss)

y_pre1 = np.array([range(100,105), range(301,306)]).T
# y_pre2 = np.array([range(201,206), range(411,416), range(150,155)]).T
# y_pre3 = np.array([range(90,95), range(110,115),range(120,125),range(130,135)]).T

y_pre = model.predict([y_pre1])

print("3101~3105 까지 나와라\n", y_pre)

# loss: 0.099999 이하 만들기

# loss :  [3.957748504035408e-06, 1.287460349885805e-06, 2.670288040462765e-06]