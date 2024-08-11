# 09_2 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15,9, 6, 17,23,21,20])

#맹그러서 그려봐!!!

# [검색] train과 test를 섞어서 7:3으로 나눠라
# 힌트 : 사이킷런
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                     train_size=0.7,      # 디폴트 0.75
                                    # test_size=0.4,
                                    # shuffle = True,    # 디폴트 True
                                      random_state=300,
                                    )

# def aaa(a, b):
#     a = a+b
#     return x_train, x_test, y_train, y_test

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)     # 사이킷런 사용법

print('x_train :', x_train)          #[2 4 8 5 1 6 3]
print('x_test : ', x_test)           #[10  7  9]
print('y_train : ', y_train)          #[2 4 8 5 1 6 3]
print('y_test : ', y_test)           #[10  7  9]

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

# 로스 :  6.753585338592529
# r2스코어 :  0.8577360922061708

#R2 평가지표 1에 가까울수록 좋다.

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)    #  y_test, y_predict 매개변수

print("RMSE : ", rmse)
