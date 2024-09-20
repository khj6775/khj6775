# 배깅은 한개의 모델, 보팅은 여러개의 모델.  배깅하면 랜덤포레스트
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 은 분류
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd

#1. 데이터
path = 'C:/AI5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)    #[652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)   # [116 rows x 1 columns]

print(train_csv.shape)  # (652, 9)
print(test_csv.shape)   # (116, 8)
print(submission_csv.shape) # (116, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 노 프라블름

# train_csv = train_csv[train_csv['BloodPressure'] > 0]
# train_csv = train_csv[train_csv['BMI'] > 0.0]

x = train_csv.drop(['Outcome'], axis=1)
# print(x)    # [652 rows x 8 columns]
y = train_csv['Outcome']

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
#                           bootstrap=True,   # 디폴트, 중복안됨.  
#                           # bootstrap=False,  # false = 중복허용 안함
                        #  )

# model = LogisticRegression()

model = XGBClassifier()


# model = CatBoostClassifier()

# model = BaggingClassifier(CatBoostClassifier(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           # bootstrap=True,   # 디폴트, 중복안됨.  
#                           bootstrap=False,  # false = 중복허용 안함
#                          )


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

# 디시전트리
# 최종점수 : 0.7251908396946565
# acc_score : 0.7251908396946565

# BaggingClassifier(DecisionTreeClassifier()), bootstrap=True
# 최종점수 : 0.7862595419847328
# acc_score : 0.7862595419847328

# BaggingClassifier(DecisionTreeClassifier()), bootstrap=False
# 최종점수 : 0.7099236641221374
# acc_score : 0.7099236641221374

# model = XGBClassifier()
# 최종점수 : 0.7633587786259542
# acc_score : 0.7633587786259542


# LogisticRegression
# 최종점수 : 0.7786259541984732
# acc_score : 0.7786259541984732

# model = CatBoostClassifier()
# 최종점수 : 0.7938931297709924
# acc_score : 0.7938931297709924