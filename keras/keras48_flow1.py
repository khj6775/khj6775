from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨와
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온거 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/ai5/_data/image/me/me.png'

img = load_img(path, target_size=(100,100),)
print(img)

# print(type(img))
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (146, 180, 3) -> (100, 100, 3) 
print(type(arr))    # <class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape)    # (1, 100, 100, 3)

######################  요기부터 증폭  #######################

datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip=True,         # 수직 뒤집기 = 증폭
    # width_shift_range=0.1,      # 평행 이동   = 증폭
    # height_shift_range=0.1,     # 평행 이동 수직
    rotation_range=15,           # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,             # 축소 또는 확대
    # shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode='nearest',        # 원래 있던 가까운 놈으로 채운다.
)

it = datagen.flow(img,               # flow = 수치화된 데이터를 변환하거나 증폭한다.
             batch_size=1,
            )

print(it)

print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))


for i in range(5):
    batch = it.next()   
    print(batch.shape)      # (1, 100, 100, 3)
    batch = batch.reshape(100,100,3)

    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()

