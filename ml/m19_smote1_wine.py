import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(33)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)  # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
x = x[:-39]
y = y[:-39]
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.75, shuffle=True, random_state=123, 
    stratify=y,
    )

'''
#2. 모델
# model = XGBClassifier()
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',     # sparse 원핫 안해도 대지롱
              metrics=['acc'])
model.fit(x_train, y_train, epochs = 100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# f1_score
y_predict = model.predict(x_test)
print(y_predict)    # 3개의 값이 나오니까 argmax해야것지!!!

y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
# [1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 2 1 1 0 1 0 1 0]

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print('acc : ', acc)
print('f1_score : ', f1)

# acc :  0.7714285714285715
# f1_score :  0.5357142857142857
'''

# exit()

###################### SMOTE 적용 ##############################
# pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk                    
print('사이킷런 : ', sk.__version__)   # 사이킷런 :  1.5.1

# 통상적으로 트레인에만 적용한다.

print('증폭전 :', np.unique(y_train, return_counts=True))

smote = SMOTE(random_state=7777)
x_train, y_train = smote.fit_resample(x_train, y_train)
print('증폭후 :', np.unique(y_train, return_counts=True))
# 증폭전 : (array([0, 1, 2]), array([44, 53,  6], dtype=int64))
# 증폭후 : (array([0, 1, 2]), array([53, 53, 53], dtype=int64))

# 스모팅 세리면 시간이 제곱으로 늘어난다. 그래서 원데이터를 분할후 스모팅한다. 그라믄 시간 절약!

######################### 스모팅 적용 끗 #########################################


#2. 모델
# model = XGBClassifier()
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',     # sparse 원핫 안해도 대지롱
              metrics=['acc'])
model.fit(x_train, y_train, epochs = 100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# f1_score
y_predict = model.predict(x_test)
print(y_predict)    # 3개의 값이 나오니까 argmax해야것지!!!

y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
# [1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 2 1 1 0 1 0 1 0]

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')

print('acc : ', acc)
print('f1_score : ', f1)


######### smote 적용 전 ########
# acc :  0.8571428571428571
# f1_score :  0.597096188747731


######### smote 적용 후 ########
# acc :  0.8857142857142857
# f1_score :  0.6259259259259259