import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 
x,y = load_digits(return_X_y=True)

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

# 최종점수 :  0.975
# acc :  0.975

print(model.feature_importances_)
# [0.0000000e+00 7.6794820e-03 9.5549468e-03 7.7860206e-03 1.0972269e-02
#  3.3153225e-02 6.1281878e-03 1.8750118e-03 1.1830009e-03 7.5491243e-03
#  2.3262659e-02 3.4366632e-03 1.0159871e-02 1.4521747e-02 8.7261060e-03
#  4.9100796e-05 4.0408919e-05 7.3477905e-03 1.2294458e-02 2.9069183e-02
#  1.5108360e-02 5.4323692e-02 1.1021708e-02 4.6551093e-05 2.4109073e-04
#  1.0252012e-02 3.0073272e-02 2.0349396e-02 1.8739015e-02 3.0457361e-02
#  1.5850088e-02 5.4323978e-06 0.0000000e+00 5.1599115e-02 1.9057905e-02
#  1.9780558e-02 5.9461523e-02 2.9630588e-02 2.1119056e-02 0.0000000e+00
#  1.9477523e-06 5.1232949e-03 3.8730826e-02 4.6639480e-02 1.8630747e-02
#  1.1262058e-02 1.7107310e-02 5.3910597e-05 0.0000000e+00 1.8761726e-03
#  1.1333466e-02 1.6362857e-02 4.6384898e-03 2.1259364e-02 3.7967447e-02
#  1.1845498e-02 0.0000000e+00 3.4216316e-03 3.0822761e-02 5.2614808e-03
#  5.8924735e-02 1.5402629e-02 3.0250674e-02 1.1177286e-02]

thresholds = np.sort(model.feature_importances_)    # 오름차순
print(thresholds)
# [0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#  1.9477523e-06 5.4323978e-06 4.0408919e-05 4.6551093e-05 4.9100796e-05
#  5.3910597e-05 2.4109073e-04 1.1830009e-03 1.8750118e-03 1.8761726e-03
#  3.4216316e-03 3.4366632e-03 4.6384898e-03 5.1232949e-03 5.2614808e-03
#  6.1281878e-03 7.3477905e-03 7.5491243e-03 7.6794820e-03 7.7860206e-03
#  8.7261060e-03 9.5549468e-03 1.0159871e-02 1.0252012e-02 1.0972269e-02
#  1.1021708e-02 1.1177286e-02 1.1262058e-02 1.1333466e-02 1.1845498e-02
#  1.2294458e-02 1.4521747e-02 1.5108360e-02 1.5402629e-02 1.5850088e-02
#  1.6362857e-02 1.7107310e-02 1.8630747e-02 1.8739015e-02 1.9057905e-02
#  1.9780558e-02 2.0349396e-02 2.1119056e-02 2.1259364e-02 2.3262659e-02
#  2.9069183e-02 2.9630588e-02 3.0073272e-02 3.0250674e-02 3.0457361e-02
#  3.0822761e-02 3.3153225e-02 3.7967447e-02 3.8730826e-02 4.6639480e-02
#  5.1599115e-02 5.4323692e-02 5.8924735e-02 5.9461523e-02]

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

# Trech=0.000, n=64, ACC: 97.50%
# Trech=0.000, n=64, ACC: 97.50%
# Trech=0.000, n=64, ACC: 97.50%
# Trech=0.000, n=64, ACC: 97.50%
# Trech=0.000, n=64, ACC: 97.50%
# Trech=0.000, n=59, ACC: 97.50%
# Trech=0.000, n=58, ACC: 97.50%
# Trech=0.000, n=57, ACC: 97.50%
# Trech=0.000, n=56, ACC: 97.50%
# Trech=0.000, n=55, ACC: 97.50%
# Trech=0.000, n=54, ACC: 97.50%
# Trech=0.000, n=53, ACC: 97.50%
# Trech=0.001, n=52, ACC: 97.50%
# Trech=0.002, n=51, ACC: 97.50%
# Trech=0.002, n=50, ACC: 97.50%
# Trech=0.003, n=49, ACC: 97.22%
# Trech=0.003, n=48, ACC: 97.50%
# Trech=0.005, n=47, ACC: 97.50%
# Trech=0.005, n=46, ACC: 97.78%
# Trech=0.005, n=45, ACC: 97.78%
# Trech=0.006, n=44, ACC: 97.78%
# Trech=0.007, n=43, ACC: 97.78%
# Trech=0.008, n=42, ACC: 97.50%
# Trech=0.008, n=41, ACC: 97.78%
# Trech=0.008, n=40, ACC: 97.78%
# Trech=0.009, n=39, ACC: 97.50%
# Trech=0.010, n=38, ACC: 97.50%
# Trech=0.010, n=37, ACC: 97.78%
# Trech=0.010, n=36, ACC: 97.78%
# Trech=0.011, n=35, ACC: 98.06%
# Trech=0.011, n=34, ACC: 97.78%
# Trech=0.011, n=33, ACC: 97.50%
# Trech=0.011, n=32, ACC: 97.50%
# Trech=0.011, n=31, ACC: 97.50%
# Trech=0.012, n=30, ACC: 97.78%
# Trech=0.012, n=29, ACC: 97.50%
# Trech=0.015, n=28, ACC: 97.22%
# Trech=0.015, n=27, ACC: 97.22%
# Trech=0.015, n=26, ACC: 97.22%
# Trech=0.016, n=25, ACC: 96.67%
# Trech=0.016, n=24, ACC: 96.67%
# Trech=0.017, n=23, ACC: 96.11%
# Trech=0.019, n=22, ACC: 96.39%
# Trech=0.019, n=21, ACC: 95.56%
# Trech=0.019, n=20, ACC: 96.11%
# Trech=0.020, n=19, ACC: 96.39%
# Trech=0.020, n=18, ACC: 95.56%
# Trech=0.021, n=17, ACC: 95.28%
# Trech=0.021, n=16, ACC: 96.11%
# Trech=0.023, n=15, ACC: 95.00%
# Trech=0.029, n=14, ACC: 95.28%
# Trech=0.030, n=13, ACC: 93.06%
# Trech=0.030, n=12, ACC: 93.89%
# Trech=0.030, n=11, ACC: 91.94%
# Trech=0.030, n=10, ACC: 91.39%
# Trech=0.031, n=9, ACC: 88.33%
# Trech=0.033, n=8, ACC: 85.83%
# Trech=0.038, n=7, ACC: 82.78%
# Trech=0.039, n=6, ACC: 76.67%
# Trech=0.047, n=5, ACC: 71.94%
# Trech=0.052, n=4, ACC: 59.17%
# Trech=0.054, n=3, ACC: 44.17%
# Trech=0.059, n=2, ACC: 35.00%
# Trech=0.059, n=1, ACC: 25.56%