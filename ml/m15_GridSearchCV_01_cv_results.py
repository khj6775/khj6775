import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

#1. 데이터
x,y = load_iris(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {"C":[1, 10, 100, 1000], 'kernel' :['linear', 'sigmoid'], 'degree':[3,4,5]},  # 24개 파라미터
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},   # 6개 파라미터
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'],
     'gamma':[0.01, 0.001, 0.0001], 'degree':[3,4]}     # 24
]   # 54

#2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold
                     , verbose=1,
                     refit=True,    # 젤 좋은 모델로 함 더 돌린다
                     n_jobs=-1,     # 모든 cpu 코어를 다 돌린다
                     )

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)  
# 최적의 매개변수 :  SVC(C=10, kernel='linear')
print('최적의 파라미터 : ', model.best_params_)     # 요놈이 degree 까지 좀 더 자세히 알려주는 듯
# 최적의 파라미터 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}
print('best_score : ', model.best_score_)   # train 만 들어가서 점수가 더 좋아
# best_score :  0.9916666666666668     
print('model.score : ', model.score(x_test, y_test)) # train, test 다 해서 점수가 더 낮다.
# model.score :  0.9
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# accuracy_score :  0.9
y_pred_best = model.best_estimator_.predict(x_test)     # 둘이 같은거니까 이걸 쓰자.
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
# 최적 튠 ACC :  0.9
print('걸린시간 : ', round(end_time - start_time, 2), '초')
# 걸린시간 :  1.48 초

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)    # 판다스 데이터도 확인가능
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # 오름차순
print(pd.DataFrame(model.cv_results_).columns)
# ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
#        'split0_test_score', 'split1_test_score', 'split2_test_score',
#        'split3_test_score', 'split4_test_score', 'mean_test_score',
#        'std_test_score', 'rank_test_score']

path = './_save/m15_GS_CV_01/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True) \
    .to_csv(path + 'm15_GS_cv_results.csv')