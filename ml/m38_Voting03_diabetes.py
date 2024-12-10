# 배깅은 한개의 모델, 보팅은 여러개의 모델.  배깅하면 랜덤포레스트
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression     # LogisticRegression 은 분류
from sklearn.ensemble import BaggingRegressor, VotingRegressor, VotingRegressor
from xgboost import XGBRegressor, XGBRegressor
from catboost import CatBoostRegressor, CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.datasets import fetch_california_housing



#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
Lgbm = LGBMRegressor()
cat = CatBoostRegressor()
rf = RandomForestRegressor()

model = XGBRegressor()
# model = VotingRegressor(
    # estimators = [('XGB', xgb), ('rf', rf), ('CAT', cat)],
    #voting = 'hard',   # 디폴트
    # voting = 'soft',
# )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)

# XGBRegressor
# 최종점수 :  0.38133437892539235
# r2_score :  0.38133437892539235

# voting
# 최종점수 :  0.46830323545904273
# r2_score :  0.46830323545904273