import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
# text = '우리 도현이랑 다희는 너무 너무 아주 그냥 이쁘다'


token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)        # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
# print(x.shape)  # 리스트는 shape 없어!!!!
print(len(x))

######### 원핫 3가지 맹그러봐!!!! ##########

# # 케라스
from tensorflow.keras.utils import to_categorical
# x = np.reshape(x, (-1,1))

x_ohe = to_categorical(x, num_classes=10)
x_ohe = x_ohe[:, :, 1:]
x_ohe = x_ohe.reshape(14,9)

print(x_ohe)
print(x_ohe.shape)

# [[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (14, 9)

# 사이킷런
from sklearn.preprocessing import OneHotEncoder     # 전처리
# x_ohe = x.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)    # True가 디폴트 
x_ohe = np.array(x).reshape(-1,1)
x_ohe = ohe.fit_transform(x_ohe)   # -1 은 데이터 수치의 끝 
                                # sklearn의 문법 = 행렬로 주세요, reshape 할때 데이터의 값과 순서가 바뀌면 안된다.
print(x_ohe)
print(x_ohe.shape)

# [[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (14, 9)

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
#     1  2  3  4  5  6  7  8  9
# 0   0  0  0  1  0  0  0  0  0
# 1   0  0  0  0  1  0  0  0  0
# 2   0  1  0  0  0  0  0  0  0
# 3   0  1  0  0  0  0  0  0  0
# 4   0  0  1  0  0  0  0  0  0
# 5   0  0  1  0  0  0  0  0  0
# 6   0  0  0  0  0  1  0  0  0
# 7   0  0  0  0  0  0  1  0  0
# 8   0  0  0  0  0  0  0  1  0
# 9   1  0  0  0  0  0  0  0  0
# 10  1  0  0  0  0  0  0  0  0
# 11  1  0  0  0  0  0  0  0  0
# 12  1  0  0  0  0  0  0  0  0
# 13  0  0  0  0  0  0  0  0  1
# (14, 9)