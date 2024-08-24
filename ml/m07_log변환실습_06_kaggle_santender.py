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