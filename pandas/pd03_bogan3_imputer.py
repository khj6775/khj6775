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

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# ImportError: IterativeImputer is experimental and the API might change without any deprecation cycle. To use it, you need to explicitly import enable_iterative_imputer:
# from sklearn.experimental import enable_iterative_imputer

imputer = SimpleImputer()       # 디폴트 평균
data2 = imputer.fit_transform(data) 
print(data2)

imputer = SimpleImputer(strategy='mean')     # 평균
data3 = imputer.fit_transform(data)  
print(data3)

imputer = SimpleImputer(strategy='median')   # 중위
data4 = imputer.fit_transform(data)  
print(data4)

imputer = SimpleImputer(strategy='most_frequent')   # 최빈값, 가장자주나오는놈
data5 = imputer.fit_transform(data)  
print(data5)

imputer = SimpleImputer(strategy='constant', fill_value=777)   # 상수, 특정값
data6 = imputer.fit_transform(data)  
print(data6)

imputer = KNNImputer()  # KNN 알고리즘으로 결측치 처리
data7 = imputer.fit_transform(data)
print(data7)

imputer = IterativeImputer()    # 선형회귀 알고리즘 !!! // MICE방식.  iterpolate 와 비슷하지만, 마지막 값을 찾아준다.
data8 = imputer.fit_transform(data)
print(data8)

