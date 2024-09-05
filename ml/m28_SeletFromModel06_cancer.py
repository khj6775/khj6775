import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=3377, train_size=0.8,
    stratify=y
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
model = XGBClassifier(
    n_estimators = 1000,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,      # L1 규제
    reg_lambda = 1,     # L2 규제
    eval_metrics='logloss',    # 2.1.1 버전에서 위로 감.
    callbacks=[early_stop],
    random_state = 3377,      # 랜덤스테이트 고정해주자
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
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# 최종점수 :  0.9912280701754386
# acc :  0.9912280701754386

print(model.feature_importances_)
# [3.1095673e-03 2.6227774e-02 3.9051243e-03 1.6615657e-05 2.0730145e-02
#  8.1776241e-03 5.2664266e-03 8.4118381e-02 7.8737978e-03 4.9598357e-03
#  1.3275798e-02 1.4882818e-03 5.9160744e-03 6.1758894e-02 1.0046216e-02
#  7.9790084e-03 9.9304272e-03 1.4152890e-02 1.5800500e-02 5.6784889e-03
#  5.2224431e-02 4.9101923e-02 7.0977546e-02 2.1681710e-01 2.2077497e-02
#  1.6301911e-02 2.9043932e-02 1.9351867e-01 1.4708905e-02 2.4816213e-02]

thresholds = np.sort(model.feature_importances_)    # 오름차순
print(thresholds)
# [1.6615657e-05 1.4882818e-03 3.1095673e-03 3.9051243e-03 4.9598357e-03
#  5.2664266e-03 5.6784889e-03 5.9160744e-03 7.8737978e-03 7.9790084e-03
#  8.1776241e-03 9.9304272e-03 1.0046216e-02 1.3275798e-02 1.4152890e-02
#  1.4708905e-02 1.5800500e-02 1.6301911e-02 2.0730145e-02 2.2077497e-02
#  2.4816213e-02 2.6227774e-02 2.9043932e-02 4.9101923e-02 5.2224431e-02
#  6.1758894e-02 7.0977546e-02 8.4118381e-02 1.9351867e-01 2.1681710e-01]

from sklearn.feature_selection import SelectFromModel

for i in thresholds: 
    seletion = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = seletion.transform(x_train)
    select_x_test = seletion.transform(x_test)

    select_model = XGBClassifier(
                                n_estimators = 1000,
                                max_depth = 6,
                                gamma = 0,
                                min_child_weight = 0,
                                subsample = 0.4,
                                reg_alpha = 0,      # L1 규제
                                reg_lambda = 1,     # L2 규제
                                eval_metrics='logloss',    # 2.1.1 버전에서 위로 감.
                                # callbacks=[early_stop],
                                random_state = 3377,)      # 랜덤스테이트 고정해주자
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_test, y_test)],
                     verbose=0
                     )
    
    selelct_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, selelct_y_predict)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))

#     Trech=0.000, n=30, ACC: 99.12%
# Trech=0.001, n=29, ACC: 99.12%
# Trech=0.003, n=28, ACC: 99.12%
# Trech=0.004, n=27, ACC: 99.12%
# Trech=0.005, n=26, ACC: 99.12%
# Trech=0.005, n=25, ACC: 99.12%
# Trech=0.006, n=24, ACC: 99.12%
# Trech=0.006, n=23, ACC: 99.12%
# Trech=0.008, n=22, ACC: 99.12%
# Trech=0.008, n=21, ACC: 99.12%
# Trech=0.008, n=20, ACC: 99.12%
# Trech=0.010, n=19, ACC: 99.12%
# Trech=0.010, n=18, ACC: 99.12%
# Trech=0.013, n=17, ACC: 99.12%
# Trech=0.014, n=16, ACC: 99.12%
# Trech=0.015, n=15, ACC: 99.12%
# Trech=0.016, n=14, ACC: 97.37%
# Trech=0.016, n=13, ACC: 99.12%
# Trech=0.021, n=12, ACC: 99.12%
# Trech=0.022, n=11, ACC: 98.25%
# Trech=0.025, n=10, ACC: 96.49%
# Trech=0.026, n=9, ACC: 97.37%
# Trech=0.029, n=8, ACC: 97.37%
# Trech=0.049, n=7, ACC: 97.37%
# Trech=0.052, n=6, ACC: 95.61%
# Trech=0.062, n=5, ACC: 93.86%
# Trech=0.071, n=4, ACC: 94.74%
# Trech=0.084, n=3, ACC: 95.61%
# Trech=0.194, n=2, ACC: 94.74%
# Trech=0.217, n=1, ACC: 92.11%