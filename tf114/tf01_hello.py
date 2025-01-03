import tensorflow as tf
print(tf.__version__)   # 1.14.0

## 텐서플로 설치 오류시...
# pip install protobuf==3.20
# pip install numpy==1.16

print('hello world')

hello = tf.constant('hello world')
print(hello)    # Tensor("Const:0", shape=(), dtype=string)
# 그냥 프린트하면 그래프의 상태가 출력된다.

sess = tf.Session()
print(sess.run(hello))  #b'hello world'
# 세션.런을 해줘야 그래프 연산값이 출력된다.