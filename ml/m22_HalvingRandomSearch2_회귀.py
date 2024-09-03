from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import time
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x,y = load_diabetes(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True, random_state=333, train_size=0.8,
    # stratify=y,
)

print(x_train.shape, y_train.shape)    # (353, 10) (353,)
print(x_test.shape, y_test.shape)      # (89, 10) (89,)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3333)

parameters = [
    {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],'max_depth' : [3,4,5,6,8],},
    # {'learning_rate' : [0.01, 0.05, 0.1],'max_depth' : [3,4,5],},

    #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample' : [0.6,0.7,0.8,0.9,1.0],},
    #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree' :[0.6,0.7,0.8,0.9,1.0],},
    #  {'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3], 'gamma' : [0, 0.1, 0.2, 0.5, 1.0],},
]   

#2. 모델
model = HalvingRandomSearchCV(XGBRegressor(
                                         tree_method='hist',
                                         device='cuda',
                                         n_estimators=200,
                                         ),
                                         parameters,
                                         cv=kfold,
                                         verbose=2,
                                         refit=True,
                                        #  n_jobs=-1,
                                        #  n_iter=10,
                                         random_state=333,
                                         factor=3,
                                         min_resources=20,
                                         max_resources=360,
                                         aggressive_elimination=True,
                                         )

start_time = time.time()
model.fit(x_train, y_train,
          verbose=False,
          eval_set=[(x_test, y_test)],
          )
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)  

print('최적의 파라미터 : ', model.best_params_)     # 요놈이 degree 까지 좀 더 자세히 알려주는 듯

print('best_score : ', model.best_score_)   # train 만 들어가서 점수가 더 좋아
    
print('model.score : ', model.score(x_test, y_test)) # train, test 다 해서 점수가 더 낮다.

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)     # 둘이 같은거니까 이걸 쓰자.
print('최적 튠 ACC : ', r2_score(y_test, y_pred_best))

print('걸린시간 : ', round(end_time - start_time, 2), '초')


# 최적의 파라미터 :  {'max_depth': 4, 'learning_rate': 0.01}
# best_score :  0.29768973250053354
# model.score :  0.400725124029488
# r2_score :  0.400725124029488
# 최적 튠 ACC :  0.400725124029488
# 걸린시간 :  78.08 초

