import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print(x.shape, y.shape) # (7, 3) (7,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1],1)
print(x.shape)
# 3-D tensor with shape (batch_size, timesteps, features)

model = load_model('./_save/keras52/LSTM/k52_0807_1737_0202-0.0059.hdf5')

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss :', results)

x_pred = np.array([50,60,70]).reshape(1,3,1)
y_pred = model.predict(x_pred)
# (3, )->(1,3,1)

print('[50,60,70]의 결과 : ', y_pred)

# x_pred = np.array([50,60,70])

# loss : 0.0005832071183249354
# [50,60,70]의 결과 :  [[80.69334]]

# loss : 0.002362962579354644
# [50,60,70]의 결과 :  [[80.10468]]

# 17:20
# loss : [0.017035122960805893, 0.0]
# [50,60,70]의 결과 :  [[80.40093]]

# k52_0807_1722_0106-0.0628.hdf5
# loss : 0.20735175907611847
# [50,60,70]의 결과 :  [[79.7879]]
