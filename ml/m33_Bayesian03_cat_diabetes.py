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
x, y = load_diabetes(return_X_y=True)

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

# {'target': 0.6184184487217004, 'params': {'bagging_temperature': 5.0, 'border_count': 172.92415123534857, 'depth': 4.0, 'l2_leaf_reg': 
# 8.471860099584267, 'learning_rate': 0.2, 'random_strength': 1.0}}
# 50 번 걸린시간 : 20.45 초