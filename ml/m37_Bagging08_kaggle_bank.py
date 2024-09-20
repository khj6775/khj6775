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
path = 'C:/AI5/_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'replaced_train.csv', index_col=0)
# print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'replaced_test.csv', index_col=0)
# print(test_csv)     # [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
# print(submission_csv)      # [110023 rows x 1 columns]

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
train_csv=train_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis = 1)
# print(x)    # [165034 rows x 10 columns]

y = train_csv['Exited']

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
#                          )

# model = LogisticRegression()

# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=False,   # 디폴트, 중복안됨.  
#                           # bootstrap=False,  # false = 중복허용 안함
#                          )

# model = RandomForestClassifier()

# model = BaggingClassifier(RandomForestClassifier(),   # 랜포도 배깅이다
#                           n_estimators=75,            # OSError: [WinError 1450] 시스템 리소스가 부족하기 때문에 요청한 서비스를 완성할 수 없습니다
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True, 
#                           # bootstrap=False,
                        #  )

# model = XGBClassifier()

# model = CatBoostClassifier()

model = BaggingClassifier(CatBoostClassifier(),   
                          n_estimators=100,            
                          n_jobs=-1,
                          random_state=4444,
                          # bootstrap=True, 
                          bootstrap=False,
                          )





#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)


# model = DecisionTreeClassifier()
# 최종점수 : 0.7953464416638895
# acc_score : 0.7953464416638895

# BaggingClassifier(DecisionTreeClassifier(), bootstrap=True 
# 최종점수 : 0.8548792680340533
# acc_score : 0.8548792680340533

# BaggingClassifier(DecisionTreeClassifier(), bootstrap=False 
# 최종점수 : 0.7992547035477323
# acc_score : 0.7992547035477323

# model = LogisticRegression()
# 최종점수 : 0.8314296967309964
# acc_score : 0.8314296967309964

# BaggingClassifier(LogisticRegression(), bootstrap=True 
# 최종점수 : 0.831369103523495
# acc_score : 0.831369103523495

# BaggingClassifier(LogisticRegression(), bootstrap=False
# 최종점수 : 0.8314296967309964
# acc_score : 0.8314296967309964

# model = RandomForestClassifier()
# 최종점수 : 0.8586663435028933
# acc_score : 0.8586663435028933

# BaggingClassifier(RandomForestClassifier(), bootstrap=True, 
# 최종점수 : 0.8604841397279365
# acc_score : 0.8604841397279365

# BaggingClassifier(RandomForestClassifier(), bootstrap=False, 
# 최종점수 : 0.8605144363316872
# acc_score : 0.8605144363316872


# model = XGBClassifier()
# 최종점수 : 0.863453206895507
# acc_score : 0.863453206895507

# model = CatBoostClassifier()
# 최종점수 : 0.8644226982155301
# acc_score : 0.8644226982155301

# BaggingClassifier(CatBoostClassifier(), bootstrap=True,
# 최종점수 : 0.864210621989275
# acc_score : 0.864210621989275

# BaggingClassifier(CatBoostClassifier(), bootstrap=False,
