from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
]

x_pre = ['태운이 참 재밋네요.']

y = labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
token1 = Tokenizer()
token1.fit_on_texts(x_pre)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, ' 
# 반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

#  {'태': 1, '운': 2, '이': 3, '참': 4, '재': 5, '미': 6, '없': 7, '다': 8}


x = token.texts_to_sequences(docs)

y_pre = token.texts_to_sequences(x_pre)

print(x)
print(type(x)) # <class 'list'>
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], 
# [10, 11, 12, 13, 14], [15], 
# [16], [17, 18], [19, 20], 
# [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]

print(y_pre)
print(type(y_pre)) # <class 'list'>
# [[1], [2], [3], [], [4], [], [5], [6], [7], [8], []]



# 변수안의 리스트의 최대 길이 찾기
'''
max_len = max(len(item) for item in x)
print('최대 길이 :',max_len)
'''

# 넘파이로 패딩
'''
for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
padded_np
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences #많이 사용할 것 같은 너낌
# from keras.utils import pad_sequences
x = pad_sequences(x,
                   # padding='pre','post'
                   maxlen=5,
                   #truncating='pre'
                   ) # pre: 앞으로 post:뒤로
# y = pad_sequences(y, maxlen=15)

y_pre = pad_sequences(y_pre,maxlen=5)

####################################################

# x_end = to_categorical(x)

# y_end = to_categorical(y_pre,num_classes=31)

# x_end = x_end.reshape()

# print(x_end, x_end.shape) # (15, 5, 31)

# print(y_end, y_end.shape) # (1, 5, 31)

####################################################

# 임베딩을 할때는 원 핫 인코딩을 하지 않아야 한다.#
print(x, x.shape)

print(y_pre, y_pre.shape)

# (xy == padded_np).all() # 결과가 같은지 확인하는 파이썬 코드

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3)

# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) # (12, 5, 31) (3, 5, 31) (12,) (3,)

# x_train = x_train.reshape(12, 5, 31)

# y_end = y_end.reshape(1, 29, 31)
# print(x_train.shape,y_train.shape)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
########################## 임베딩1 ##################################################
# model.add(Embedding(input_dim=31,output_dim=100, input_length=5))   # (None, 5, 100)
# input_dim 은 단어 사전의 개수
# output_dim 다음 레이어로 갈 노드의 임의의 개수
# input_length = 열
# param의 갯수는 input_dim * output_dim이다.
# output_shape가 (none, 0, 0)으로 출력 됨으로 Conv1D나 RNN계열의 레이어를 사용할 수 있다.

# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________



########################### 임베딩2 ########################################
# model.add(Embedding(input_dim=31, output_dim=100))
# input_length 를 안넣어도 돌아간다. 자동 임의 완성

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 10)                4440

#  dense (Dense)               (None, 10)                110

#  dense_1 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 7,661
# Trainable params: 7,661
# Non-trainable params: 0
# _________________________________________________________________

########################### 임베딩3 ########################################
# model.add(Embedding(input_dim=10, output_dim=100))
# input_dim 의 숫자를 다르게 넣어도 돌아간다. 성능이 다를 수 있다. 숫자가 적으면 부족, 많으면 낭비

########################### 임베딩4 ########################################
model.add(Embedding(31,100))                      # 잘 돌아가
# model.add(Embedding(31,100,5))                  # 에러
# model.add(Embedding(31,100, input_length=5))    # 잘 돌아가
# model.add(Embedding(31,100, input_length=1))    # 잘 돌아가는데 약수만 가능

model.add(LSTM(10))     #  (None, 10)
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=100)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss', results)

pred = model.predict(y_pre)
print('태운이 참 재밋네요.', np.round(pred))

 