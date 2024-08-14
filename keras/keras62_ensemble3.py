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
x3_datasets = np.array([range(100), range(301, 401),
                        range(77,177), range(33,133)]).T
                        # 거시기1, 거시기2, 거시기3, 거시기4

y1 = np.array(range(3001, 3101))  # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))  # 비트코인 가격.

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, \
    y2_train, y2_test = train_test_split(                                   
    x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.8, random_state= 215)

print(x1_train.shape, x1_test)
print(x2_train.shape, x2_test)
print(x3_train.shape, x3_test)
print(y1_train.shape, y1_test)
print(y2_train.shape)
# exit()

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(32, activation='relu', name='bit1')(input1)
dense2 = Dense(64, activation='relu', name='bit2')(dense1)
dense3 = Dense(32, activation='relu', name='bit3')(dense2)
dense4 = Dense(16, activation='relu', name='bit4')(dense3)
output1 = Dense(8, activation='relu', name='bi5')(dense4)
model1 = Model(inputs=input1, outputs=output1)
model1.summary()

#2-2. 모델
input11 = Input(shape=(3,))
dense11 = Dense(32, activation='relu', name='bit11')(input11)
dense21 = Dense(16, activation='relu', name='bit21')(dense11)
output11 = Dense(8, activation='relu', name='bit31')(dense21)
model2 = Model(inputs=input11, outputs=output11)

#2-3. 모델
input2 = Input(shape=(4,))
dense111 = Dense(32, activation='relu', name='bit110')(input2)
dense211 = Dense(16, activation='relu', name='bit210')(dense111)
output111 = Dense(8, activation='relu', name='bit310')(dense211)
model3 = Model(inputs=input2, outputs=output111)



#2-4. 합체!!!
# merge1 = concatenate([output1, output11], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output11, output111])
merge2 = Dense(4, name='mg2')(merge1)
merge3 = Dense(2, name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)   # y 값 두개로 나눈다
last_output2 = Dense(1, name='last2')(merge3)

# middle_output = Dense(1, name='last')(merge3)

# #1 model = Model(inputs=[input1, input11, input2], outputs=[last_output1, last_output2])
# #2 model = Model(inputs=[input1, input11, input2], outputs=middle_output)
# # model.summary()

# # 2-5. 분기1.
# dense31 = Dense(64, activation='relu', name='bit41')(middle_output)
# dense32 = Dense(32, activation='relu', name='bit42')(dense31)
# last_output1 = Dense(1, name='bit43')(dense32)

# # 2-6. 분기2.
# dense33 = Dense(128, activation='relu', name='bit44')(middle_output)
# dense34 = Dense(64, activation='relu', name='bit45')(dense33)
# last_output2 = Dense(1, name='bit46')(dense34)

# 2-7. 다시 합체

model = Model(inputs=[input1, input11, input2], outputs=[last_output1, last_output2])

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=5000, batch_size=128)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

path = './_save/keras62/ensemble3/'
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

model.fit([x1_train, x2_train, x3_train], [y1_train,y2_train],
          epochs=3000,
          batch_size=1024,
          validation_split=0.2,
          callbacks=[es,mcp])


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test], verbose=1)
print('loss : ', loss)

y_pre1 = np.array([range(100,105), range(301,306)]).T
y_pre2 = np.array([range(201,206), range(411,416), range(150,155)]).T
y_pre3 = np.array([range(90,95), range(110,115),range(120,125),range(130,135)]).T

y_pre = model.predict([y_pre1, y_pre2, y_pre3])

print("3101~3105 까지 나와라\n", y_pre)

# loss 3개 출력이유 = [y1+y2, y1, y2]

# loss :  [9.566545031702844e-07, 9.834766245830906e-08, 8.583068620282575e-07]