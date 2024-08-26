from sklearn.datasets import fetch_california_housing
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
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()    # population 이거 이상해
# df.plot.box()
# plt.show()

# print(df.info())
# print(df.describe())

# df['Population']. boxplot()    # 시리즈에서 이거 안돼
# df['Population'].plot.box()      # 이거 돼
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

###################### x의 Population 로그 변환 ##############################
x['Population'] = np.log1p(x['Population'])  # 지수변환 np.expm1

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size = 0.8, random_state=333)

#################### y 로그 변환 #########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##########################################################

#2. 모델구성
# model = RandomForestRegressor(random_state=1234,
#                               max_depth=5,
#                               min_samples_split=3,
#                               )

model = LinearRegression()


# model = Sequential()
# model.add(Dense(16, input_dim=8))
# model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', 
#                     patience=30, verbose=0,
#                     restore_best_weights=True,
#                     )


model.fit(x_train, y_train
        #   , epochs=300, batch_size=32, callbacks=[es]
        )

#4. 평가, 예측
score = model.score(x_test, y_test)
print('score : ', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print(r2)

# RF
# score :  0.6476965541046513

# x,y 둘다 로그 변환
# score :  0.6633724443363114


#LR
# y만 로그변환
# score :  0.6190645225752244