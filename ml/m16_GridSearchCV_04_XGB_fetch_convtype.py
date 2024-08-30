from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold
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
x, y = fetch_covtype(return_X_y=True)     # sklearn에서 데이터를 x,y 로 바로 반환

# one hot encoding
y = np.argmax(pd.get_dummies(y).values, axis=1)
# pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

parameters =[
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6,10,12],
     'min_samples_leaf' : [3, 10]},  #12
    {'n_jobs' : [-1,], 'max_depth' : [6,8,10,12],
     'min_samples_leaf' : [3,5,7, 10]}, # 16
     {'n_jobs' : [-1,], 'min_samples_leaf' : [3,5,7,10],
     'min_samples_split' : [2,3,5,10]}, # 16
     {'n_jobs' : [-1,], 'min_samples_leaf' : [2,3,5,10]},   # 4
]   # 48

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

#2. 모델
model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=kfold
                     , verbose=1,
                     refit=True,    # 젤 좋은 모델로 함 더 돌린다
                     n_jobs=-1,     # 모든 cpu 코어를 다 돌린다
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

