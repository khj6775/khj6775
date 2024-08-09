from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))    # (4, 4, 10)
# 5,5,1 = 가로, 세로, 색깔       # 이미지는 4차원 = 가로,세로,색깔,n장
model.add(Conv2D(5,(2,2)))                           # (3, 3, 5)

model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50

#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0