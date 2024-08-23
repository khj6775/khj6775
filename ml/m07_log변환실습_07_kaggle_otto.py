import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_digits
import numpy as np
import time

#1. 데이터
path = 'C:/AI5/_data/kaggle/Otto Group/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [61878 rows x 94 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [144368 rows x 93 columns]

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission_csv)   # [144368 rows x 9 columns]

print(train_csv.columns)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']
print(y.shape)

y_ohe = pd.get_dummies(y)

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
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
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
# score :  0.4638738128108102

# x만 로그변환
# score :  

# y만 로그변환
# score :  0.4270903441575822

# x,y 둘다 로그 변환
# score : 