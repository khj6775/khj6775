from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.svm import SVC
import numpy as np
import time
import xgboost as xgb
import warnings
# warnings.filterwarnings('ignore')

x,y = load_diabetes(return_X_y = True)

# datasets = load_diabetes()
# df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# df['target'] = datasets.target

# x = df.drop(['target'], axis=1).copy()
# y = df['target']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    # stratify=y,
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
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)

#2. 모델
model = GridSearchCV(xgb.XGBRegressor(), parameters, cv=kfold
                     , verbose=1,
                     refit=True,    # 젤 좋은 모델로 함 더 돌린다
                     n_jobs=-1,     # 모든 cpu 코어를 다 돌린다
                     )

start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)  

print('최적의 파라미터 : ', model.best_params_)     # 요놈이 degree 까지 좀 더 자세히 알려주는 듯

print('best_score : ', model.best_score_)   # train 만 들어가서 점수가 더 좋아

print('model.score : ', model.score(x_test, y_test)) # train, test 다 해서 점수가 더 낮다.

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)     # 둘이 같은거니까 이걸 쓰자.
print('최적 튠 r2 : ', r2_score(y_test, y_pred_best))

print('걸린시간 : ', round(end_time - start_time, 2), '초')

# 최적의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=8, max_leaves=None,
#              min_child_weight=None, min_samples_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=None,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 3, 'n_jobs': -1}
# best_score :  0.30290445736242977
# model.score :  0.2514538207330046
# r2_score :  0.2514538207330046
# 최적 튠 r2 :  0.2514538207330046
# 걸린시간 :  6.34 초