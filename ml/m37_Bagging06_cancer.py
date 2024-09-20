# 배깅은 한개의 모델, 보팅은 여러개의 모델.  배깅하면 랜덤포레스트
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression     # LogisticRegression 은 분류
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=4444, train_size=0.8,
    stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                         #   bootstrap=True,   # 디폴트, 중복안됨.  
#                           bootstrap=False,  # false = 중복허용 안함
                        #  )

# model = LogisticRegression()

model = CatBoostClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

# 디시전트리
# 최종점수 : 0.9473684210526315
# acc_score : 0.9473684210526315

# BaggingClassifier(DecisionTreeClassifier()), bootstrap=False
# 최종점수 : 0.9649122807017544
# acc_score : 0.9649122807017544

# LogisticRegression
# 최종점수 : 0.9736842105263158
# acc_score : 0.9736842105263158