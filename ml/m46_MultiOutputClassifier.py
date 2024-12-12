import numpy as np
import pandas as pd

from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

np.random.seed(42)


# 다중분류 데이터 생성 함수
def create_multiclass_data_with_labels():
    # X 데이터 생성 (20, 3)
    X = np.random.rand(20, 3)

    # y 데이터 생성 (20, 3)
    y = np.random.randint(0, 5, size=(20, 3))  # 각 클래스에 0부터 9까지 값

    # 데이터프레임으로 변환
    X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Label2', 'Label3'])

    return X_df, y_df

X, y = create_multiclass_data_with_labels()
print("X 데이터:")
print(X)
print("\nY 데이터:")
print(y)

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

#2. 모델
model = RandomForestClassifier()
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) 
print(model.predict([[0.195983,  0.045227,  0.325330]]))  
# RandomForestClassifier 스코어 :  0.0
# [[4 2 4]]


model = MultiOutputClassifier(XGBClassifier())
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  
print(model.predict([[0.195983,  0.045227,  0.325330]]))  
# MultiOutputClassifier 스코어 :  0.0333
# [[4 2 4]]

model = MultiOutputClassifier(LGBMClassifier())
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  
print(model.predict([[0.195983,  0.045227,  0.325330]])) 
# MultiOutputClassifier 스코어 :  1.6
# [[2 0 4]]


model = MultiOutputClassifier(CatBoostClassifier())
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred.shape) #(1,20,3)
y_pred = y_pred.reshape(20, 3)

print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  
print(model.predict([[0.195983,  0.045227,  0.325330]])) 

# MultiOutputClassifier 스코어 :  0.0
# [[[4 2 4]]]



# y가 다차원으로 나왔을 때 사용하세요 MultiOutputRegressor
# catboost 랑  LGBM 은 다차원이 안댐. 멀티아웃풋 래핑 고고