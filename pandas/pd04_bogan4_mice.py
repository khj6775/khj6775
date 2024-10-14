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

np.float = float    # numpy 버전 안바꾸고 mice 적용
print(np.__version__)   #1.22.4 에서 실행, 2.1.1에서 에러
# pip install impyute
from impyute.imputation.cs import mice
data9 = mice(data.values,
             n=10,
             seed=777)
print(data9)