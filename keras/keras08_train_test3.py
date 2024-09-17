import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split          # 사이킷런 추가
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 나눠라
# 힌트 : 사이킷런
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                     train_size=0.7,      # 디폴트 0.75
                                    # test_size=0.4,
                                    # shuffle = True,    # 디폴트 True
                                    #  random_state=1004,
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
model.add(Dense (1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 : ', loss)
print('[11]의 예측값: ', results)

# 로스 :  0.0137212676927447
# [11]의 예측값:  [[11.155632]]