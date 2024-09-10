import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1.데이터
x, y = load_wine(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8,
    stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves,  min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),   # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),
        'reg_lambda' : max(reg_lambda, 0),
        'reg_alpha' : reg_alpha,
    }

    model = XGBClassifier(**params, n_jobs=-1)

    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metrics='logloss',
              verbose=0,
            )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 150
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)      # 제일 좋은거 보여준다. 데이터에만 신뢰함.
print(n_iter, '번 걸린시간 :', round(end_time-start_time, 2), '초')

# {'target': 1.0, 'params': {'colsample_bytree': 0.6053181530964207, 'learning_rate': 0.00948017959891945, 'max_bin': 9.0, 'max_depth': 3.40723813508719, 'min_child_samples': 128.60144563141185, 'min_child_weight': 1.0, 'num_leaves': 26.042411129702824, 'reg_alpha': 0.01, 'reg_lambda': 1.3159178931865392, 'subsample': 0.9172777855103045}}
# 150 번 걸린시간 : 69.68 초