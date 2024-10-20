import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan],])
# print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer        # iterpolate 와 비슷하지만, 마지막 값을 찾아준다.
from sklearn.impute import IterativeImputer
# ImportError: IterativeImputer is experimental and the API might change without any deprecation cycle. To use it, you need to explicitly import enable_iterative_imputer:
# from sklearn.experimental import enable_iterative_imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor

imputer = IterativeImputer()     # 디폴트 BayesianRidge 회귀모델.
data1 = imputer.fit_transform(data)
print(data1)

imputer = IterativeImputer(estimator=DecisionTreeRegressor())
data2 = imputer.fit_transform(data)
print(data2)

imputer = IterativeImputer(estimator=RandomForestRegressor())
data3 = imputer.fit_transform(data)
print(data3)

# imputer = IterativeImputer(estimator=XGBRegressor())
# data4 = imputer.fit_transform(data)
# print(data4)
