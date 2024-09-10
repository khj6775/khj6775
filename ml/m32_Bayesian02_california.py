import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1.데이터
x, y = fetch_california_housing(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, train_size=0.8,
    # stratify=y,
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

    import xgboost as xgb
    early_stop = xgb.callback.EarlyStopping(
    rounds=30,
    # metric_name='mlogloss',  # error
    data_name='validation_0',
    save_best=True,
    )

    model = XGBRegressor(**params, n_jobs=-1,
                            callbacks=[early_stop],
                         )

    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metrics='logloss',
              verbose=0,
            )
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)      # 제일 좋은거 보여준다. 데이터에만 신뢰함.
print(n_iter, '번 걸린시간 :', round(end_time-start_time, 2), '초')

# {'target': 0.8535445741909286, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 500.0, 'max_depth': 10.0, 'min_child_samples': 88.85911959000515, 'min_child_weight': 20.524239942555578, 'num_leaves': 24.163988738039777, 'reg_alpha': 0.01, 'reg_lambda': 7.740650496182449, 'subsample': 1.0}}
# 300 번 걸린시간 : 123.33 초
