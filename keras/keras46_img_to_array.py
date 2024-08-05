from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온거 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/ai5/_data/image/me/me.png'

img = load_img(path, target_size=(100,100),)
print(img)

print(type(img))
plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (146, 180, 3) -> (100, 100, 3) 
print(type(arr))    # <class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape)    # (1, 100, 100, 3)

print

np_path = 'c:/AI5/_data/image/me/'     # 수치들의 형태는 다 넘파이다.
np.save(np_path + 'keras46_image_me.npy', arr=img)
