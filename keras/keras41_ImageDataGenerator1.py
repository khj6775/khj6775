import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip=True,         # 수직 뒤집기 = 증폭
    width_shift_range=0.1,      # 평행 이동   = 증폭
    height_shift_range=0.1,     # 평행 이동 수직
    rotation_range=5,           # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,             # 축소 또는 확대
    shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode='nearest',        # 원래 있던 가까운 놈으로 채운다.
)

test_datagen = ImageDataGenerator(
    rescale=1./255,             # test 데이터는 rescale만 하고 절대 건들지 않는다!!
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(       # 디렉토리로 부터 흘러가듯 가져오세요
    path_train,
    target_size=(200, 200),                         # 이미지 사이즈 통일
    batch_size=10,                                  # (10,200,200,1)
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)   # Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(       # 디렉토리로 부터 흘러가듯 가져오세요
    path_test,
    target_size=(200, 200),                        # 이미지 사이즈 통일
    batch_size=10,                                 # (10,200,200,1)
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,                                # test는 shuffle 안한다.
    # Found 120 images belonging to 2 classes.
)

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000236457F5700>
print(xy_train.next())          # x 와 y 가 모여있는 Iterator
print(xy_train.next())
# for i in Iterator

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][1])
# print(xy_train[0].shape)      # AttributeError: 'tuple' object has no attribute 'shape'
print(xy_train[0][0].shape)     # (10, 200, 200, 1)
# print(xy_train[16])            # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[15][2])         # IndexError: tuple index out of range

print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'>
print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>














