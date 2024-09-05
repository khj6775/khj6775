from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets)

df = pd.DataFrame(x, columns=datasets.feature_names)
# print(df)
df['Target'] = y
print(df)

print("================================== 상관계수 히트맵 =====================================")
print(df.corr())

# =================================== 상관계수 히트맵 =================================================
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
# Target                      0.782561         -0.426658           0.949035          0.956547  1.000000

# 역시나 과적합이 생길 수 있다. 높은 점수를 삭제해도 성능이 나아 질 수 있다.

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)          #0.11.2 
print(matplotlib.__version__)   #3.4.3
# sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,         # 표안에 수치 명시
            cbar=True)          # 사이드 바
plt.show()

