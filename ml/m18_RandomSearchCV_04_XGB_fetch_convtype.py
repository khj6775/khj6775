from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x = scaler.fit_transform(x)
# x_test = scaler.transform(x_test)
# one hot encoding
y = pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

parameters =[
   {'n_jobs' : [-1,], 'n_estimators' : [100,250, 500], 'max_depth' : [4,8,10,12],
     'min_samples_leaf' : [3,5, 10], 'tree_method' : ['gpu_hist'], 'learning_rate' : [0.002] }, # 36
    {'n_jobs' : [-1,], 'max_depth' : [6,10,12], 'learning_rate' : [0.003,0.008],
     'min_samples_leaf' : [3,5,9,10],  'tree_method' : ['gpu_hist']},  # 24
     {'n_jobs' : [-1,], 'min_samples_leaf' : [3,7,9,10], 'learning_rate' : [0.004,0.007],
     'min_samples_split' : [3,6,8,10], 'tree_method' : ['gpu_hist']}, # 32
     {'n_jobs' : [-1,], 'min_samples_leaf' : [2,4,6,8,10], 'tree_method' : ['gpu_hist'], 'learning_rate' : [0.005,0.006,0.009]}, # 15
]   # 48

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)

#2. 모델
model = RandomizedSearchCV(xgb.XGBClassifier(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,    # 젤 좋은 모델로 함 더 돌린다
                     n_jobs=-1,     # 모든 cpu 코어를 다 돌린다
                     n_iter=10,  # 요걸로 search candidate 수 조절
                     random_state=321,
                     )

start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True
          )

end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)  

print('최적의 파라미터 : ', model.best_params_)     # 요놈이 degree 까지 좀 더 자세히 알려주는 듯

print('best_score : ', model.best_score_)   # train 만 들어가서 점수가 더 좋아

print('model.score : ', model.score(x_test, y_test)) # train, test 다 해서 점수가 더 낮다.

y_predict = model.predict(x_test)
print('acc_score : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)     # 둘이 같은거니까 이걸 쓰자.
print('최적 튠 acc : ', accuracy_score(y_test, y_pred_best))

print('걸린시간 : ', round(end_time - start_time, 2), '초')
