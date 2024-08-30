from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = 'C:/AI5/_data/kaggle/Otto Group/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [61878 rows x 94 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [144368 rows x 93 columns]

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission_csv)   # [144368 rows x 9 columns]

# print(train_csv.columns)

# print(train_csv.isna().sum())   # 0
# print(test_csv.isna().sum())    # 0

# label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])
# print(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']
print(y.shape)

y_ohe = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

parameters =[
   {'n_jobs' : [-1,], 'n_estimators' : [100, 250, 500], 'max_depth' : [4,6,8,10,12],
     'min_samples_leaf' : [3,5, 10], 'tree_method' : ['gpu_hist'], 'learning_rate' : [0.002] },
    {'n_jobs' : [-1,], 'max_depth' : [6,8,10,12], 'learning_rate' : [0.003],
     'min_samples_leaf' : [3,5,7,9, 10],  'tree_method' : ['gpu_hist']}, 
     {'n_jobs' : [-1,], 'min_samples_leaf' : [3,5,7,10], 'learning_rate' : [0.004],
     'min_samples_split' : [2,3,5,8,10], 'tree_method' : ['gpu_hist']},
     {'n_jobs' : [-1,], 'min_samples_leaf' : [2,3,5,8,10], 'tree_method' : ['gpu_hist'], 'learning_rate' : [0.005]},
]   # 48

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)

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

# 최적의 매개변수 :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.002, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=12, max_leaves=None,
#               min_child_weight=None, min_samples_leaf=10, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 :  {'tree_method': 'gpu_hist', 'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 10, 'max_depth': 12, 'learning_rate': 0.002}
# best_score :  0.7830998501324731
# model.score :  0.7835326438267615
# acc_score :  0.7835326438267615
# 최적 튠 acc :  0.7835326438267615
# 걸린시간 :  819.9 초