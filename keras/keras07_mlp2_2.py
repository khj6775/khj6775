import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array(range(10))
print(x)    #[0 1 2 3 4 5 6 7 8 9]             # range 사용 x=np.array(range(10)) 0~9
print(x.shape)  #(10,)

x = np.array(range(1, 11))
print(x)    #[ 1  2  3  4  5  6  7  8  9 10]
print(x.shape)  #(10,)

x = np.array([range(10), range(21,31), range(201,211)])
x=x.T
print(x)    #[[  0  21 201]
#  [  1  22 202]
#  [  2  23 203]
#  [  3  24 204]
#  [  4  25 205]
#  [  5  26 206]
#  [  6  27 207]
#  [  7  28 208]
#  [  8  29 209]
#  [  9  30 210]]
#[[  0   1   2   3   4   5   6   7   8   9]
            # [ 21  22  23  24  25  26  27  28  29  30]
            # [201 202 203 204 205 206 207 208 209 210]]
print(x.shape)  #(3,10)

y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습]
#[10, 31, 211] 예측할것

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('로스 :', loss)
print('[10,31,211]의 예측값 :', results)

# 로스 : 0.0757221356034278
# [10,31,211]의 예측값 : [[11.020445]]