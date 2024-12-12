import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

#1. 데이터
x, y = load_linnerud(return_X_y=True)
print(x.shape, y.shape)     # (20, 3) (20, 3)
print(x)
print(y)
########## 요론 데이터얌 ############
# [  5. 162.  60.] -> [191.  36.  50.]
# ................
# [  2. 110.  43.] -> [138.  33.  68.]

#2. 모델
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # RandomForestRegressor 스코어 : 3.6102
print(model.predict([[2, 110, 43]]))    # [[155.76  34.34   63.22]]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # LinearRegression 스코어 :  7.4567
print(model.predict([[2, 110, 43]]))    # [[187.33745435  37.08997099  55.40216714]]

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4)) # Ridge 스코어 :  7.4569
print(model.predict([[2, 110, 43]]))    # [[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # XGBRegressor 스코어 :  0.0008
print(model.predict([[2, 110, 43]]))    # [[138.00215   33.001656  67.99831 ]]

# model = CatBoostRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 : ',
#       round(mean_absolute_error(y, y_pred), 4))  
# print(model.predict([[2, 110, 43]])) 

# error : Currently only multi-regression, multilabel and survival objectives work with multidimensional target


# model = LGBMRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 : ',
#       round(mean_absolute_error(y, y_pred), 4))  
# print(model.predict([[2, 110, 43]])) 

# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.


from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  
print(model.predict([[2, 110, 43]])) 
# MultiOutputRegressor 스코어 :  8.91
# [[178.6  35.4  56.1]]

model = MultiOutputRegressor(CatBoostRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  
print(model.predict([[2, 110, 43]])) 
# MultiOutputRegressor 스코어 :  0.2154
# [[138.97756017  33.09066774  67.61547996]]


model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  
print(model.predict([[2, 110, 43]])) 
# CatBoostRegressor 스코어 :  0.0638
# [[138.21649371  32.99740595  67.8741709 ]]


# y가 다차원으로 나왔을 때 사용하세요 MultiOutputRegressor
# catboost 랑  LGBM 은 다차원이 안댐. 멀티아웃풋 래핑 고고