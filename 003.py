import joy
import numpy as np

g = lambda x : 1 / (1+np.exp(-x))

(X, y) = joy.loads_mnist_num(8)
W1 = joy.load_mnit_weight('./w_xh.weights')
Z1 = np.dot(W1, X)
A1 = g(Z1)

W2 = joy.load_mnist_weight('./w_xh.weights')
Z2 = np.dot(W2, A1)
yhat = g(Z2)

print(y)
print(np.round_(yhat, 3))



import tensorflow as tf
import numpy as np

# Sigmoid 함수 정의
g = lambda x: 1 / (1 + np.exp(-x))

# MNIST 데이터 로드
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 7번 숫자에 해당하는 데이터 필터링
X = X_test[y_test == 9]
y = y_test[y_test == 9]

# 가중치 초기화 (임의로 설정된 가중치, 실제 모델을 학습시키면 더 좋습니다)
W1 = np.random.randn(128, 784)  # 입력층(784) -> 은닉층(128) 가중치
W2 = np.random.randn(10, 128)   # 은닉층(128) -> 출력층(10) 가중치

# 입력 데이터 전처리 (28x28 이미지를 784개의 피처로 변환)
X = X.reshape(X.shape[0], -1)  # (n_samples, 784)
X = X / 255.0  # 정규화

# 은닉층 계산
Z1 = np.dot(W1, X.T)
A1 = g(Z1)

# 출력층 계산
Z2 = np.dot(W2, A1)
yhat = g(Z2)

print(y)
print(np.round_(yhat.T, 3).argmax(axis=1))
