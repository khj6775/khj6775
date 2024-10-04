import tensorflow as tf
print("텐서플로 버전 : ", tf.__version__)

if tf.config.list_physical_devices('GPU'):
    print('GPU 있다!!!')
else :
    print('GPU 없다!!!')

# Cuda 버전
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print('CuDA 버전 : ', cuda_version)

# CuDNN 버전
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print('CuDNN 버전 : ', cudnn_version)