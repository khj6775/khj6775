# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = 'C:/AI5/_data/kaglle/dogs-vs-cats-redux-kernels-edition/train/'
path_test = 'C:/AI5/_data/kaglle/dogs-vs-cats-redux-kernels-edition/test/'
path_sub = 'C:/AI5/_data/kaglle/dogs-vs-cats-redux-kernels-edition/'

submission_csv = pd.read_csv(path_sub + "sample_submission.csv", index_col=0)

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
)
x = xy_train[0][0]
y = xy_train[0][1]


x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=9843,
)

end = time.time()


#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(100, 100, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.summary()

         # 필터로 증폭, 커널 사이즈로 자른다.                              
                                # shape = (batch_size, height, width, channels), (batch_size, rows, columns, channels)   
                                # shape = (batch_size, new_height, new_width, filters)
                                # batch_size 나누어서 훈련한다
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=16, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(1, activation='sigmoid'))


model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=160,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)
# r2 = r2_score(y_test, y_pre)
# print('r2 score :', r2)
# y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)

print('걸린시간 : ', end-start, '초')

y_submit = np.round(model.predict(xy_test))

submission_csv['label'] = y_submit

submission_csv.to_csv(path_sub + "submission_0802_1705.csv")


# loss : 0.5869136452674866
# acc : 0.7744

# loss : 0.4627073109149933
# acc : 0.786