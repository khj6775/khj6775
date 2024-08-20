# 15개의 행에서
# 5개를 더 넣어서(2개는 6개 이상 단어, turncate 사용)
# 예) 반장 주말에 출근 혜지 안혜지 안혜지 //0
# 맹그러
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Conv1D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다', 
    '반장 주말에 출근 혜지 안혜지 안혜지', '밥을 맛있게 냠냠 쩝쩝 냠냠 쩝쩝 또 냠냠 먹어요',
    '아따 날 덥구마잉', '태풍이 온대요', '곧 가을이 와요'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '냠냠': 2, '너무': 3, '반장': 4, '또': 5, '안혜지': 6, '쩝쩝': 7, '재미있다': 8, '최고에요': 9, '잘만든': 10, '영화예요': 11, '추천하
# 고': 12, '싶은': 13, '영화입니다': 14, '한': 15, '번': 16, '더': 17, '보고': 18, '싶어요': 19, '글쎄': 20, '별로에요': 21, '생각보다': 22, '지 
# 루해요': 23, '연기가': 24, '어색해요': 25, '재미없어요': 26, '재미없다': 27, '재밋네요': 28, '준영이': 29, '바보': 30, '잘생겼다': 31, '태운이': 32, '구라친다': 33, '주말에': 34, '출근': 35, '혜지': 36, '밥을': 37, '맛있게': 38, '먹어요': 39, '아따': 40, '날': 41, '덥구마잉': 42, '태풍
# 이': 43, '온대요': 44, '곧': 45, '가을이': 46, '와요': 47}

x = token.texts_to_sequences(docs)
print(x)
# [[3, 8], [1, 9], [1, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19], [20], [21], [22, 23], 
#  [24, 25], [26], [3, 27], [1, 28], [29, 30], [4, 31], [32, 5, 33], [4, 34, 35, 36, 6, 6], 
#  [37, 38, 2, 7, 2, 7, 5, 2, 39], [40, 41, 42], [43, 44], [45, 46, 47]]
print(type(x))  # <class 'list'>

### 패딩 ###
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                        # padding = 'pre', 'post',         # (디폴트) 앞에 0 / 뒤에 0 채우기
                        maxlen = 9,                      # n개로 자르기, 앞에가 잘림 
                        truncating = 'pre' # , 'post'      # 앞에서 / 뒤에서 부터 자르기                         
                         )
print(padded_x)
print(padded_x.shape)   # (20, 9)

# exit()
padded_x = padded_x.reshape(20,9,1)

##ohe
from tensorflow.keras.utils import to_categorical
padded_x = to_categorical(padded_x)
# padded_x = padded_x[:, :, 1:]
print(padded_x.shape)  # (20, 9, 48)

x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, test_size=0.1, random_state=755)

#2. 모델 구성 
model = Sequential()
model.add(Conv1D(16, 3, input_shape=(9,48)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras65/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath = "".join([path, 'k65_06_', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1, 
        #   validation_split=0.1,
        #   callbacks=[es] #, mcp],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

x_pre = ["곧 가을이 와요"]
x_pre = token.texts_to_sequences(x_pre)
x_pre = pad_sequences(x_pre, maxlen = 9)
print(x_pre.shape)  # (1, 9)

x_pre = to_categorical(x_pre, num_classes=48)
x_pre = x_pre.reshape(1,9,48)
print(x_pre.shape)  # (1, 9, 48)

y_pre = model.predict(x_pre)
print("곧가을이와요 : ", np.round(y_pre))

print("걸린 시간 :", round(end-start,2),'초')

# loss : 2.5991172790527344
# acc : 0.5
# 태운이 참 재미없다 의 결과 : [[1.]]
# 걸린 시간 : 5.3 초
