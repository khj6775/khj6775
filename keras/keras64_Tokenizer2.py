import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Concatenate, concatenate

import pandas as pd

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다.'

# 맹그러봐.!!

token = Tokenizer()
token.fit_on_texts([text1,text2])

print(token.word_index)
#{'마구': 1, '진짜': 2, '매우': 3, '못생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, '태운이는': 11, '선생
# 을': 12, '괴롭힌다': 13, '준영이는': 14, '사영이는': 15, '더': 16}
print(token.word_counts)
#OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 6), ('먹었다', 1), ('태운 
# 이는', 1), ('선생을', 1), ('괴롭힌다', 1), ('준영이는', 1), ('못생겼다', 2), ('사영이는', 1), ('더', 1)])

x1 = token.texts_to_sequences([text1])
x2 = token.texts_to_sequences([text2])

print(x1)    # [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10]]
print(x2)    # [[11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)

x = concatenate([x1,x2], axis=1)

# # 케라스
from tensorflow.keras.utils import to_categorical

x = to_categorical(x)
x = x[:, :, 1:]
x = x.reshape(24,16)

print(x)
print(x.shape)

# 사이킷런
from sklearn.preprocessing import OneHotEncoder     # 전처리
# x_ohe = x.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)    # True가 디폴트 
x_ohe = np.array(x).reshape(-1,1)
x_ohe = ohe.fit_transform(x_ohe)   # -1 은 데이터 수치의 끝 
                                # sklearn의 문법 = 행렬로 주세요, reshape 할때 데이터의 값과 순서가 바뀌면 안된다.
print(x_ohe)
print(x_ohe.shape)

# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# (24, 16)

# 판다스
import pandas as pd

# x= np.array(x)
# x=x.reshape(-1,)
x = pd.get_dummies(np.array(x).reshape(-1,)) 
print(x)
# x = pd.DataFrame(x)
# x_ohe = pd.get_dummies(x) 
# print(x_ohe)
print(x.shape)

#     1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16
# 0    0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
# 1    0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0
# 2    0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 3    0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 4    0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
# 5    0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
# 6    0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
# 7    0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
# 8    0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0
# 9    1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 10   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 11   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 12   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 13   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0
# 14   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0
# 15   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
# 16   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0
# 17   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0
# 18   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0
# 19   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0
# 20   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 21   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# 22   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1
# 23   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0
# (24, 16)