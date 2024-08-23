# CNN -> DNN

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))   # 0 ~ 99


##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

##### OHE
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


from sklearn.decomposition import PCA

pca = PCA(n_components=x_train.shape[1])  
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_     # 설명가능한 변화율

evr_cumsum = np.cumsum(evr)     #누적합
print(evr_cumsum)

print('0.95 이상 :', np.argmax(evr_cumsum>=0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum >= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum >= 0.999)+1)
print('1.0 일 때 :', np.argmax(evr_cumsum)+1)

# 0.95 이상 : 202
# 0.99 이상 : 659
# 0.999 이상 : 1481
# 1.0 일 때 : 3072

num = [202, 659, 1481, 3072]

for i in range(len(num)): 
    pca = PCA(n_components=num[i])   # 4개의 컬럼이 3개로 바뀜
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)
    
    #2. 모델 구성
    model = Sequential()
    model.add(Dense(1024, input_shape=(num[i],)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(100, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=0,
                    restore_best_weights=True,
                    )

    ###### mcp 세이브 파일명 만들기 ######
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/ml05/17_cifar100/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml05_', str(i+1), '_', date, '_', filename])   
    #####################################

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=0,     
        save_best_only=True,   
        filepath=filepath, 
    )

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=2000, batch_size=64,  
            verbose=0, 
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)

    y_pre = model.predict(x_test1, verbose=0)

    r2 = r2_score(y_test, y_pre)  
    
    print('결과', i+1)
    print('PCA :',num[i])
    print('loss :', round(loss[0],8))
    print('acc :', round(loss[1],8))
    print('r2 score :', r2) 
    print("걸린 시간 :", round(end-start,2),'초')
    print("===============================================")
    

"""
loss : 2.9738142490386963
acc : 0.27
accuracy_score : 0.273
걸린 시간 : 180.53 초

[BatchNormalization]
loss : 2.424513101577759
acc : 0.38
accuracy_score : 0.3829
걸린 시간 : 173.66 초

[stride, padding]
loss : 2.734889268875122
acc : 0.32
accuracy_score : 0.316
걸린 시간 : 215.21 초

[Max Pooling]
loss : 2.576554775238037
acc : 0.35z
accuracy_score : 0.3479
걸린 시간 : 164.28 초

[[DNN]]
loss : 3.536623001098633
acc : 0.19
accuracy_score : 0.1861
걸린 시간 : 182.54 초

"""

# 결과 1
# PCA : 202
# loss : 3.36694145
# acc : 0.2025
# r2 score : 0.09868955795185076
# 걸린 시간 : 46.82 초
# ===============================================
# 결과 2
# PCA : 659
# loss : 3.49784493
# acc : 0.2079
# r2 score : 0.09554557096602466
# 걸린 시간 : 57.36 초
# ===============================================
# 결과 3
# PCA : 1481
# loss : 3.58208561
# acc : 0.1672
# r2 score : 0.07004475657949051
# 걸린 시간 : 44.06 초
# ===============================================
# 결과 4
# PCA : 3072
# loss : 3.6111083
# acc : 0.1627
# r2 score : 0.06752432415396276
# 걸린 시간 : 81.56 초
# ===============================================
