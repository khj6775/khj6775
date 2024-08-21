from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import r2_score
import tensorflow as tf
import random as rn
rn.seed(337)    # 파이선의 랜덤 함수 seed 고정
tf.random.set_seed(337)
np.random.seed(337)     # 넘파이는 요롷게

path = 'C:/AI5/_data/kaggle/santander_customer/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [200000 rows x 200 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [200000 rows x 1 columns]

print(train_csv.shape)      # (200000, 201)
print(test_csv.shape)       # (200000, 200)
print(submission_csv.shape) # (200000, 1)

print(train_csv.columns)

# train_csv.info()    
# test_csv.info()     


x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']         # 'count' 컬럼만 넣어주세요
print(y.shape)   


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75,
    random_state=337,
)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(16))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1, verbose=0,
          batch_size=512,
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))


# =================1. 기본출력 ========================
# lr : 0.1, 로스 :0.07617604732513428
# lr : 0.1, r2 : 0.16075279200912718
# =================1. 기본출력 ========================
# lr : 0.01, 로스 :0.07719902694225311
# lr : 0.01, r2 : 0.1494822785171378
# =================1. 기본출력 ========================
# lr : 0.005, 로스 :0.07589413225650787
# lr : 0.005, r2 : 0.16385824241816038
# =================1. 기본출력 ========================
# lr : 0.001, 로스 :0.07524994760751724
# lr : 0.001, r2 : 0.17095608383360683
# =================1. 기본출력 ========================
# lr : 0.0005, 로스 :0.07512787729501724
# lr : 0.0005, r2 : 0.17229961805539784
# =================1. 기본출력 ========================
# lr : 0.0001, 로스 :0.07502568513154984
# lr : 0.0001, r2 : 0.17342615649171245