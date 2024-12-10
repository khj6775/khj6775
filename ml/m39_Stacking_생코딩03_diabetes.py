# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾아
# 스태킹 : 모델을 돌려 나온거로 컬럼을 구성해서 새로운 데이터셋을 만들.
        # 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼 맹그러


# 배깅은 한개의 모델, 보팅은 여러개의 모델.  배깅하면 랜덤포레스트
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression     # LogisticRegression 은 분류
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from xgboost import XGBRegressor, XGBRegressor
from catboost import CatBoostRegressor, CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1199, train_size=0.8,
    # stratify=y,
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# (455, 30) (114, 30)
# (455,) (114,)

# exit()

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

train_list = []
test_list = []
models = [xgb, rf, cat]
for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    train_list.append(y_predict)
    test_list.append(y_test_predict)

    score = r2_score(y_test, y_test_predict)
    class_name = model.__class__.__name__
    print('{0} R2 : {1:.4f}'.format(class_name, score))

# print(np.__version__)   # 1.22.4

x_train_new = np.array(train_list).T
# print(x_train_new.shape)   

x_test_new = np.array(test_list).T
# print(x_test_new.shape)    

#2. 모델
model2 = CatBoostRegressor(verbose=0)
model2.fit(x_train_new, y_train)
y_pred = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred)
print("스태킹결과 : ", score2)

# XGBRegressor R2 : 0.4131
# RandomForestRegressor R2 : 0.5050
# CatBoostRegressor R2 : 0.4877
# 스태킹결과 :  0.4443727244628619
