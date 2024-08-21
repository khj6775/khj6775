import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import numpy as np
import time

#1. 데이터
path = 'C:/AI5/_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'replaced_train.csv', index_col=0)
print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'replaced_test.csv', index_col=0)
print(test_csv)     # [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [110023 rows x 1 columns]

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(submission_csv.shape) # (110023, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 결측치 없음

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
train_csv=train_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis = 1)
print(x)    # [165034 rows x 10 columns]

y = train_csv['Exited']
print(y.shape)      # (165034,)


print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(np.max(x), np.min(x))     # 1.0 0.0

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
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
# 결과 1
# PCA : 8
# time : 813.2 초
# r2 score : 0.31608659926181315
# 결과 2
# PCA : 10
# time : 418.93 초
# r2 score : 0.40811465834062455
# 결과 3
# PCA : 10
# time : 400.34 초
# r2 score : 0.4057957812266778
# 결과 4
# PCA : 10
# time : 347.07 초
# r2 score : 0.4051884394155505