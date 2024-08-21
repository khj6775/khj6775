import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
import numpy as np
import time

#1. 데이터
path = "C:/AI5/_data/dacon/따릉이/"        # 경로지정  상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # . = 루트 = AI5 폴더,  index_col=0 첫번째 열은 데이터가 아니라고 해줘
print(train_csv)     # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       #[715 rows x 1 columns],    NaN = 빠진 데이터
# 항상 오타, 경로 , shape 조심 확인 주의

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape)     #(715, 1)

print(train_csv.columns)
# # ndex(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# x 는 결측치가 있어도 괜찮지만 y 는 있으면 안된다

train_csv.info()

################### 결측치 처리 1. 삭제 ###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())   # 결측치 확인

train_csv = train_csv.dropna()   # 결측치 삭제
print(train_csv.isna().sum())    # 삭제 뒤 결측치 확인
print(train_csv)        #[1328 rows x 10 columns]
print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())
#  test_csv 는 결측치 삭제 불가, test_csv 715 와 submission 715 가 같아야 한다.
#  그래서 결측치 삭제하지 않고, 데이터의 평균 값을 넣어준다.

test_csv = test_csv.fillna(test_csv.mean())     #컬럼끼리만 평균을 낸다
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)           # drop = 컬럼 하나를 삭제할 수 있다.   # axis=1 이면 열, 0 이면 행  카운트 열을 지워라
print(x)        #[1328 rows x 9 columns]
y = train_csv['count']         # 'count' 컬럼만 넣어주세요
print(y.shape)   # (1328,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(np.max(x), np.min(x))     # 1.0 0.0

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
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
    hist = model.fit(x_train1, y_train, epochs=100, batch_size=10, verbose=0, validation_split=0.2, callbacks=[es, mcp])
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
# PCA : 7
# time : 10.44 초
# r2 score : 0.6260069453522521
# 결과 2
# PCA : 9
# time : 3.33 초
# r2 score : 0.5796400570776321
# 결과 3
# PCA : 9
# time : 17.9 초
# r2 score : 0.6764217450726526
# 결과 4
# PCA : 9
# time : 11.79 초
# r2 score : 0.6224833913627319