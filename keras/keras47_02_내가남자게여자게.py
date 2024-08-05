
import numpy as np
from tensorflow.keras.models import load_model

me=np.load('C:\\AI5\\_data\\image\\me\\keras46_image_me.npy')

model = load_model('C:/AI5/_save/keras45/07_save_npy_gender/k45_gender_0805_1515_0007-0.5562.hdf5')

y_pred = np.round(model.predict(me))

print(y_pred)

if y_pred >=0.5:
    print(y_pred,'%의 확률로 여자')

else:
    print(abs(1-y_pred),"% 의 확률로 남자")
