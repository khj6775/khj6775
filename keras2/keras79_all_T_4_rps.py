# 01. VGG19
# 02. Xception
# 03. ResNet50
# 04. ResNet101
# 05. InceptionV3
# 06. InceptionResNetV2
# 07. DenseNet121
# 08. MobileNetV2
# 09. NasNetMobile
# 10. EfficeintNetB0

# GAP 써라!!!
# 기존거와 최고 성능 비교!!!

# keras79_all_T_2_cifar100

from tensorflow.keras.applications import VGG16 , VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

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

model_list = [
                VGG19,
              Xception,
              ResNet50,
              ResNet101,
              InceptionV3,
              InceptionResNetV2,
              DenseNet121,
              MobileNetV2,
            #   NASNetMobile,
              EfficientNetB0]

for i in model_list:
    models = i(# weights = 'imagenet'
                include_top = False,
               input_shape = (100,100,3)
               )

    models.trainable = False

    model = Sequential()
    model.add(models)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    # model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(3, activation='softmax'))

    # 훈련, 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=1, batch_size=64,
            verbose=1, 
            validation_split=0.2,
            callbacks=[es],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    print('loss :', loss[0])
    print('acc :', round(loss[1],2))

    y_pre = model.predict(x_test)

    y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)

    acc = accuracy_score(y_test1, y_pre)
    print('accuracy_score :', i, acc)
    print("걸린 시간 :", round(end-start,2),'초')