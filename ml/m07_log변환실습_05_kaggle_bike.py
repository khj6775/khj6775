import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
import numpy as np
import time


#1. 데이터
path = 'C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\'      # 절대경로
# path = 'C://AI5//_data//bike-sharing-demand//'      # 절대경로   다 가능
# path = 'C:/AI5/_data/bike-sharing-demand/'      # 절대경로
#  /  //  \  \\ 다 가능

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)


print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(sampleSubmission.shape)   # (6493, 1)
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count

print(train_csv.columns)       # 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe().T)   # describe 평균,중위값 등등 나타냄. 많이쓴다.

############### 결측치 확인 #################
print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())             # 전부 결측치 없음 확인
print(test_csv.isnull().sum())

############# x와 y를 분리 ########
x = train_csv.drop(['casual', 'registered','count'], axis=1)    # 대괄호 하나 = 리스트    두개 이상은 리스트
print(x)            # [10886 rows x 8 columns]

y = train_csv['count']
print(y.shape)      #(10886,)

import matplotlib.pyplot as plt

# train_csv.boxplot()    
# train_csv.plot.box()
# plt.show()

# exit()

# print(df.info())
# print(df.describe())

# exit()
# df['Population']. boxplot()    # 시리즈에서 이거 안돼
# df['Population'].plot.box()      # 이거 돼
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()


# x = df.drop(['target'], axis=1).copy()
# y = df['target']

###################### x의 Population 로그 변환 ##############################
# x['humidity'] = np.log1p(x['humidity'])  # 지수변환 np.expm1
# x['ZN'] = np.log1p(x['ZN'])  # 지수변환 np.expm1
# x['B'] = np.log1p(x['B'])  # 지수변환 np.expm1

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size = 0.8, random_state=333)

#################### y 로그 변환 #########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##########################################################

# 2. 모델구성
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,
                              )

# model = RandomForestRegressor()

model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('score : ', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print(r2)

# RF

# 안변환
# score : 0.31197696450992274

# x만 로그변환
# score :  0.31204891387050804

# y만 로그변환
# score :  0.2800319056588215

# x,y 둘다 로그 변환
# score : 0.28005211231608174