### 판다스로 바꿔서 컬럼 삭제 ####
# pd.DataFrame
# 컬럼명 : datasets.feature_names 안에 있지!!!

# 실습
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 데이터셋 재구성후
# 기존 모델결과와 비교!!!

# 끗

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x = pd.DataFrame(x, columns=datasets.feature_names)

print(x)

x = x.drop(['sepal width (cm)'], axis=1) 

random_state = 3

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    stratify=y,
)

#2. 모델구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

print('random_state :', random_state)
for model in models:
    model.fit(x_train, y_train)
    print("=============", model.__class__.__name__, "================")   # 클래스 이름만 나오게 한다.
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)



# random_state : 3
# ============= DecisionTreeClassifier ================
# acc 0.9
# [0.         0.03       0.04191995 0.92808005]
# ============= RandomForestClassifier ================
# acc 0.8666666666666667
# [0.12409436 0.0345686  0.36912916 0.47220787]
# ============= GradientBoostingClassifier ================
# acc 0.9
# [0.00267854 0.01393039 0.18616255 0.79722852]
# ============= XGBClassifier ================
# acc 0.9333333333333333
# [0.00675549 0.01673217 0.44314435 0.53336793]