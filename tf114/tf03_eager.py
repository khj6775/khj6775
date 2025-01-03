import tensorflow as tf
print('tf version : ', tf.__version__)      # tf version :  1.14.0
print('즉시실행모드 : ', tf.executing_eagerly())    # 즉시실행모드 :  False
# 세스런이 빠지고 즉시실행모드

# 즉시실행모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.
# tf.compat.v1.disable_eager_execution()  # # eager 모드 끄기 // 즉시실행모드 끈다. // 텐서플로 1.0 문법 //디폴트
# tf.compat.v1.enable_eager_execution()   # eager 모드 켜기 // 즉시실행모드 켠다. // 텐서플로 2.0 사용가능

print('즉시실행모드 : ', tf.executing_eagerly())    # False

# 텐서플로 버전을 바꿀 수 없을 때 eager 모드 전환하여 사용!! 

hello = tf.constant('Hello world')

sess = tf.compat.v1.Session()
print(sess.run(hello))

#   가상환경   즉시실행모드        사용가능
#   1.14.0     disable(디폴트)     b'Hello world'
#   1.14.0     enable              RuntimeError: The Session graph is empty.
#   2.7.4      disable(디폴트)     b'Hello world'
#   2.7.4      enable              RuntimeError: The Session graph is empty.

'''
Tensor 1 은 '그래프연산' 모드
Tensor 2 는 '즉시실행' 모드

tf.copat.v1.enable_eager_execution()    # 즉시실행모드 켜
               -> Tensor 2 의 디폴트

tf.compat.v1.disable_eager_execution()  # 즉시 실행모드 꺼
                                                -> 그래프 연산모드로 돌아간다.
                                                -> Tensor 1 코드를 쓸 수 있다.

tf.executing_eagerly() # True면 즉시실행모드, -> Tensor 2 코드만 써야한다.
                        False면 그래프 연산모드 -> Tensor 1 코드를 쓸 수 있다.
'''

