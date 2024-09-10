import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier, CatBoostRegressor


#1.데이터
path = "C:/AI5/_data/dacon/따릉이/"        # 경로지정  상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # . = 루트 = AI5 폴더,  index_col=0 첫번째 열은 데이터가 아니라고 해줘
print(train_csv)     # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       #[715 rows x 1 columns],    NaN = 빠진 데이터
# 항상 오타, 경로 , shape 조심 확인 주의

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape)     #(715, 1)

print(train_csv.columns)
# # ndex(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# x 는 결측치가 있어도 괜찮지만 y 는 있으면 안된다

train_csv.info()

################### 결측치 처리 1. 삭제 ###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())   # 결측치 확인

train_csv = train_csv.dropna()   # 결측치 삭제
print(train_csv.isna().sum())    # 삭제 뒤 결측치 확인
print(train_csv)        #[1328 rows x 10 columns]
print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())
#  test_csv 는 결측치 삭제 불가, test_csv 715 와 submission 715 가 같아야 한다.
#  그래서 결측치 삭제하지 않고, 데이터의 평균 값을 넣어준다.

test_csv = test_csv.fillna(test_csv.mean())     #컬럼끼리만 평균을 낸다
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)           # drop = 컬럼 하나를 삭제할 수 있다.   # axis=1 이면 열, 0 이면 행  카운트 열을 지워라
print(x)        #[1328 rows x 9 columns]
y = train_csv['count']         # 'count' 컬럼만 넣어주세요
print(y.shape)   # (1328,)

random_state=5
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate' : [0.01, 0.2],
    'depth' : [4, 12],
    'l2_leaf_reg' : [1, 10],
    'bagging_temperature' : [0.0, 5.0],      # 랜덤서치할때 써라.
    'border_count' : [32,255],
    'random_strength' : [1, 10],
}

def cat_hamsu(learning_rate, depth, l2_leaf_reg, bagging_temperature, border_count, random_strength):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)),   # 무조건 정수형
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        'bagging_temperature' : bagging_temperature,
        'border_count' : int(round(border_count)),
        'random_strength' : int(round(random_strength)),
    }

    cat_features = list(range(x_train.shape[1]))

    model = CatBoostRegressor(**params,)

    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metrics='logloss',
              verbose=0,
            )
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f = cat_hamsu,
    pbounds=bayesian_params,
    random_state=42,
)

n_iter = 50
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)      # 제일 좋은거 보여준다. 데이터에만 신뢰함.
print(n_iter, '번 걸린시간 :', round(end_time-start_time, 2), '초')

# {'target': 0.8038403052999139, 'params': {'bagging_temperature': 0.0, 'border_count': 155.5343707373495, 'depth': 12.0, 'l2_leaf_reg': 
# 1.0, 'learning_rate': 0.2, 'random_strength': 1.0}}
# 50 번 걸린시간 : 31.33 초