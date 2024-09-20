# 배깅은 한개의 모델, 보팅은 여러개의 모델.  배깅하면 랜덤포레스트
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 은 분류
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeRegressor()
# model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                         #   bootstrap=True,   # 디폴트, 중복안됨.  
#                           bootstrap=False,  # false = 중복허용 안함
#                          )

# model = LinearRegression()

# model = XGBRegressor()

model = CatBoostRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print('acc_score :', r2_score)

# 디시전트리
# 최종점수 : 0.6122350989994487
# acc_score : 0.6122350989994487

# BaggingClassifier(DecisionTreeClassifier()), bootstrap=False


# model = LinearRegression()
# 최종점수 : 0.6011614863584488
# acc_score : 0.6011614863584488


# model = XGBRegressor()
# 최종점수 : 0.8355445807273904
# acc_score : 0.8355445807273904


# model = CatBoostRegressor()
# 최종점수 : 0.8556294497294631
# acc_score : 0.8556294497294631