import numpy as np

def perceptron_train(X, y, epochs, eta):
    # 가중치 초기화
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        for xi, target in zip(X, y):
            update = eta * (target - predict(w, xi))
            w += update * xi
    return w

def predict(w, xi):
    return 1 if np.dot(w, xi) >= 0 else 0

def load_data(file_path):
    # joydata.txt를 읽어오는 함수
    data = np.loadtxt(file_path, delimiter='\t')
    X = data[:, :-1]  # 특성
    y = data[:, -1]   # 레이블
    return X, y

# 주어진 하이퍼파라미터를 테스트하는 함수
def evaluate_perceptron(file_path, epochs, eta, random_seed):
    np.random.seed(random_seed)
    
    X, y = load_data(file_path)
    
    # 데이터 셔플링
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Train/Test split (70% train, 30% test)
    split_index = int(len(X) * 0.7)
    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
    y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]
    
    # x0 = 1 추가
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Bias term
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]    # Bias term
    
    # 가중치 학습
    w = perceptron_train(X_train, y_train, epochs, eta)
    
    # Train과 Test 자료에 대한 예측
    train_predictions = [predict(w, xi) for xi in X_train]
    test_predictions = [predict(w, xi) for xi in X_test]
    
    # Train 및 Test 정확도 평가
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    return train_accuracy, test_accuracy

# 각 하이퍼파라미터 조합을 테스트
file_path = 'C:\\Users\\khj6775\\Downloads\\joydata.csv'
results = {}
hyperparams = [
    (2, 0.02, 5),
    (2, 0.03, 8),
    (2, 0.04, 10),
    (2, 0.05, 10)
]

for epochs, eta, seed in hyperparams:
    train_accuracy, test_accuracy = evaluate_perceptron(file_path, epochs, eta, seed)
    results[(epochs, eta, seed)] = (train_accuracy, test_accuracy)

# 결과 출력
for params, (train_acc, test_acc) in results.items():
    print(f"Epochs: {params[0]}, Eta: {params[1]}, Seed: {params[2]} -> Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
