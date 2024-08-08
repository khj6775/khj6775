import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T

print(a.shape)

size = 6    # 타입스탭 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)      # append = 리스트 뒤에 붙인다.
    return np.array(aaa)

bbb = split_x(a, size)
print('---------------------------')
print(bbb)
print('---------------------------')
print(bbb.shape)  

x = bbb[:, :-1]
y = bbb[:, -1,0]
print('---------------------------')
print(x,y)
print(x.shape, y.shape)     
# exit()

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))   # 행무시 열우선 행7 뺌
model.add(LSTM(units=10, activation='relu', input_shape=(5,2)))   # 통상적으로 LSTM 많이쓴다.
# model.add(GRU(units=10, activation='relu', input_shape=(3,1)))   

# 데이터가 커질수록 성능이 좋아진다.

model.add(Dense(16,activation='relu'))     # RNN은 바로 Dense 로 연결이 가능하다. 
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=300, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x,y, 
          epochs=1000, 
          batch_size=16, 
          validation_split=0.2,
        #   callbacks=[es]
)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_pred = np.array([[[6,4],[7, 3],[8, 2],[9, 1],[10, 0]]])
y_pred = model.predict(x_pred)

print('11나와라 : ', y_pred)

# loss : 0.00261609791778028
# 11나와라 :  [[10.474601]]