import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
   
)

test_datagen = ImageDataGenerator(
    rescale=1./255,             # test 데이터는 rescale만 하고 절대 건들지 않는다!!
)

path_train = './_data/image/rps'
# path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(       # 디렉토리로 부터 흘러가듯 가져오세요
    path_train,
    target_size=(100, 100),                         # 이미지 사이즈 통일
    batch_size=10000,                                  # (10,200,200,1)
    class_mode='categorical',                     # 다중분류 - 원핫도 되서 나와욤
    # class_mode='sparse',                            # 다중분류
    # class_mode='binary',                            # 이진분류
    # class_mode='none',                            # y값 없다!
    
    # color_mode='grayscale',
    color_mode='rgb',

    shuffle=True,
)   # Found 160 images belonging to 2 classes

# print(xy_train[0][1])

# print(xy_train[0][0].shape)


y = xy_train[0][1]

x_train, x_test, y_train, y_test = train_test_split(
            xy_train[0][0],
            y,
            train_size=0.8,
            stratify=y,
            random_state=9843
)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (3,3), 
                 activation='relu', 
                 strides=1,padding='same',
                 input_shape=(100, 100, 3)))   # 데이터의 개수(n장)는 input_shape 에서 생략, 3차원 가로세로컬러  27,27,10
model.add(MaxPooling2D())
model.add(Conv2D(filters=64,strides=1,padding='same', kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.3))         # 필터로 증폭, 커널 사이즈로 자른다.                              
model.add(Conv2D(64, (2,2), activation='relu',strides=1,padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), strides=1,padding='same',activation='relu')) 
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
model.add(Dense(3, activation='softmax'))


model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=160,
          verbose=1,
          validation_split=0.1,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)

# loss : 0.013397577218711376
# acc : 0.998
# acc_score : 0.998015873015873






