from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
]

y = labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, 
# '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)    # [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
print(type(x))  # <class 'list'>

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x,
                      maxlen=5,
                    #   truncating='post'
                      )    # value=0, padding='pre or post', maxlen=5 최대길이
print(pad_x)
print(pad_x.shape)

# pad_x = pd.DataFrame(pad_x)
# y = pd.DataFrame(y)

# x = pd.get_dummies(np.array(x).reshape(-1,)) 
# y = pd.get_dummies(np.array(y).reshape(-1,)) 

pad_x = pad_x.reshape(15,5,1)

print(pad_x.shape)

# exit()

x_train, x_test, y_train, y_test = train_test_split(pad_x,y, train_size=0.8,
                                                      random_state=315
)

#2. 모델
############### DNN 맹그러봐 ################
model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(5, 1)))
model.add(Conv1D(10, 2))

model.add(Flatten())

model.add(Dense(32))   # input_shape 는 벡터형태로  # 이미지 input_shape=(8,8,1)
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))                    # 0.3 = 30퍼센트 빼고 훈련할거에요
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 200,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=8,
                #  validation_split=0.2,
                 callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test,
                      verbose=1)
print('로스 :', loss[0])
print('acc :', round(loss[1],3))

x_pred = ["태운이 참 재미없다."]
token.fit_on_texts(x_pred)

x_pred = token.texts_to_sequences(x_pred)

x_pred = pad_sequences(x_pred,
                      maxlen=5,
                    #   truncating='post'
                      )    # value=0, padding='pre or post', maxlen=5 최대길이


y_pred = model.predict(x_pred)
y_pred = np.round(y_pred)
print(y_pred)

# accuracy_score = accuracy_score(x_test, y_test)

# print('acc_score :', accuracy_score)
# print('걸린시간 :', round(end - start , 2), '초')
