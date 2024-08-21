import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_digits
import numpy as np
import time

#1. 데이터
path = 'C:/AI5/_data/kaggle/santander_customer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [200000 rows x 200 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [200000 rows x 1 columns]

print(train_csv.shape)      # (200000, 201)
print(test_csv.shape)       # (200000, 200)
print(submission_csv.shape) # (200000, 1)

print(train_csv.columns)

# train_csv.info()    
# test_csv.info()     


x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']         # 'count' 컬럼만 넣어주세요
print(y.shape)   


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(np.max(x), np.min(x))     # 1.0 0.0

from sklearn.decomposition import PCA
pca = PCA(n_components=200)
x = pca.fit_transform(x)

evr_cumsum = np.cumsum(pca.explained_variance_ratio_)

print('0.95 :', np.argmax(evr_cumsum>=0.95)+1)      
print('0.99 :', np.argmax(evr_cumsum>=0.99)+1)      
print('0.999 :', np.argmax(evr_cumsum>=0.999)+1)    
print('1.0 :', np.argmax(evr_cumsum)+1)       

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

num = [np.argmax(evr_cumsum>=0.95)+1, np.argmax(evr_cumsum>=0.99)+1, 
       np.argmax(evr_cumsum>=0.999)+1, np.argmax(evr_cumsum)+1]
results = []

from sklearn.decomposition import PCA
for i in range(0, len(num), 1):
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델구성
    model = Sequential()
    model.add(Dense(64, input_shape=(num[i],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    #3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, restore_best_weights=True,)

    ########## mcp 세이브 파일명 만들기 시작 ##########
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = './_save/m04/'
    filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
    filepath = "".join([path, 'm05_01_date_', date, '_epo_', filename])

    ########## mcp 세이브 파일명 만들기 끝 ##########

    mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=filepath)

    start = time.time()
    hist = model.fit(x_train1, y_train, epochs=10, batch_size=10, verbose=0, validation_split=0.2, callbacks=[es, mcp])
    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=0)
    # print("loss : ", loss)

    y_predict1 = model.predict(x_test1)
    r2 = r2_score(y_test, y_predict1)

    ##### print #####
    print('결과', i+1)
    print('PCA :', num[i])
    print('time :', round(end-start,2),'초')
    print('r2 score :', r2)

