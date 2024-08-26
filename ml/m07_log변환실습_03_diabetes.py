from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression

#1. 데이터
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()    # population 이거 이상해
# df.plot.box()
# plt.show()

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

# exit()

x = df.drop(['target'], axis=1).copy()
y = df['target']

###################### x의 Population 로그 변환 ##############################
# x['CRIM'] = np.log1p(x['CRIM'])  # 지수변환 np.expm1
# x['ZN'] = np.log1p(x['ZN'])  # 지수변환 np.expm1
# x['B'] = np.log1p(x['B'])  # 지수변환 np.expm1

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size = 0.8, random_state=333)

#################### y 로그 변환 #########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##########################################################

# 2. 모델구성
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
# score : 0.39775663277051343

# x만 로그변환
# score :  

# y만 로그변환
# score :  0.4030610594913886

# x,y 둘다 로그 변환
# score : 