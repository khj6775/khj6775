import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
x,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=11, train_size=0.8,
    # stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    # metric_name='logloss',  # error     이진:logloss, 다중:mlogloss
    data_name='validation_0',
    save_best=True,
)

#2. 모델
model = XGBRegressor(
    n_estimators = 1000,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,      # L1 규제
    reg_lambda = 1,     # L2 규제
    eval_metrics='logloss',    # 2.1.1 버전에서 위로 감.
    callbacks=[early_stop],
    random_state = 11,      # 랜덤스테이트 고정해주자
)

#3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],
        #   eval_metrics='mlogloss',     # 2.1.1 버전에서 위로 감.
          verbose=1,)

#4. 평가, 예측
results = model.score(x_test,y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('acc : ', acc)

# 최종점수 :  0.545470626237589
# acc :  0.545470626237589

print(model.feature_importances_)
# [0.04934222 0.06421341 0.20518881 0.09137528 0.07692715 0.06253953
#  0.093003   0.07267798 0.18399277 0.10073981]

thresholds = np.sort(model.feature_importances_)    # 오름차순
print(thresholds)
# [0.04934222 0.06253953 0.06421341 0.07267798 0.07692715 0.09137528
#  0.093003   0.10073981 0.18399277 0.20518881]

from sklearn.feature_selection import SelectFromModel

for i in thresholds: 
    seletion = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = seletion.transform(x_train)
    select_x_test = seletion.transform(x_test)

    select_model = XGBRegressor(
                                n_estimators = 1000,
                                max_depth = 6,
                                gamma = 0,
                                min_child_weight = 0,
                                subsample = 0.4,
                                # reg_alpha = 0,      # L1 규제
                                # reg_lambda = 1,     # L2 규제
                                eval_metrics='logloss',    # 2.1.1 버전에서 위로 감.
                                # callbacks=[early_stop],
                                random_state = 9,)      # 랜덤스테이트 고정해주자
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_test, y_test)],
                     verbose=0
                     )
    
    selelct_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test, selelct_y_predict)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))

# Trech=0.049, n=10, ACC: 32.41%
# Trech=0.063, n=9, ACC: 34.65%
# Trech=0.064, n=8, ACC: 44.69%
# Trech=0.073, n=7, ACC: 39.66%
# Trech=0.077, n=6, ACC: 33.56%
# Trech=0.091, n=5, ACC: 29.69%
# Trech=0.093, n=4, ACC: 27.79%
# Trech=0.101, n=3, ACC: 36.74%
# Trech=0.184, n=2, ACC: 21.10%
# Trech=0.205, n=1, ACC: 13.04%