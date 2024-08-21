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

path = 'C:/AI5/_data/kaggle/Otto Group/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [61878 rows x 94 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)     # [144368 rows x 93 columns]

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission_csv)   # [144368 rows x 9 columns]

print(train_csv.columns)

print(train_csv.isna().sum())   # 0
print(test_csv.isna().sum())    # 0

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
# lr : 0.1, 로스 :3.098365306854248
# lr : 0.1, r2 : 0.5041888214967833
# =================1. 기본출력 ========================
# lr : 0.01, 로스 :3.012179136276245
# lr : 0.01, r2 : 0.5179802084562266
# =================1. 기본출력 ========================
# lr : 0.005, 로스 :2.9857449531555176
# lr : 0.005, r2 : 0.5222106831052254
# =================1. 기본출력 ========================
# lr : 0.001, 로스 :2.9840922355651855
# lr : 0.001, r2 : 0.5224754105809921
# =================1. 기본출력 ========================
# lr : 0.0005, 로스 :2.9844963550567627
# lr : 0.0005, r2 : 0.5224105555086771
# =================1. 기본출력 ========================
# lr : 0.0001, 로스 :2.9835476875305176
# lr : 0.0001, r2 : 0.5225622263820836