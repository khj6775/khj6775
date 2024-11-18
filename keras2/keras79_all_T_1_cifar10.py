# 01. VGG19
# 02. Xception
# 03. ResNet50
# 04. ResNet101
# 05. InceptionV3
# 06. InceptionResNetV2
# 07. DenseNet121
# 08. MobileNetV2
# 09. NASNetMobile
# 10. EfficientNetB0

# GAP 써라!!!
# 기존거와 최고 성능 비교!!!

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
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = tf.image.resize(x_train, (75,75)).numpy()
x_test = tf.image.resize(x_test, (75,75)).numpy()

##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.


model_list = [
                # VGG19,Xception,ResNet50,ResNet101,
            #   InceptionV3,
            #   InceptionResNetV2,DenseNet121,
            #   MobileNetV2,
            #   NASNetMobile,
              EfficientNetB0
            ]

for i in model_list:
    models = i(# weights = 'imagenet'
                include_top = False,
            #    input_shape = (224,224,3)
               )

    models.trainable = False

    model = Sequential()
    model.add(models)
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    # model.add(Dense(32))
    model.add(Dense(10))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=1, batch_size=32,
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

