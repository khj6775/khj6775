import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
x,y = load_wine(return_X_y=True)

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

# 최종점수 :  0.9722222222222222
# acc :  0.9722222222222222

print(model.feature_importances_)
# [0.08128835 0.05061164 0.02086868 0.03936667 0.11409903 0.03518124
#  0.19918582 0.         0.         0.18040927 0.12124164 0.01048258
#  0.14726512]

thresholds = np.sort(model.feature_importances_)    # 오름차순
print(thresholds)
# [0.         0.         0.01048258 0.02086868 0.03518124 0.03936667
#  0.05061164 0.08128835 0.11409903 0.12124164 0.14726512 0.18040927
#  0.19918582]

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

# Trech=0.000, n=13, ACC: 97.22%
# Trech=0.000, n=13, ACC: 97.22%
# Trech=0.010, n=11, ACC: 97.22%
# Trech=0.021, n=10, ACC: 97.22%
# Trech=0.035, n=9, ACC: 94.44%
# Trech=0.039, n=8, ACC: 94.44%
# Trech=0.051, n=7, ACC: 94.44%
# Trech=0.081, n=6, ACC: 97.22%
# Trech=0.114, n=5, ACC: 97.22%
# Trech=0.121, n=4, ACC: 97.22%
# Trech=0.147, n=3, ACC: 97.22%
# Trech=0.180, n=2, ACC: 97.22%
# Trech=0.199, n=1, ACC: 66.67%