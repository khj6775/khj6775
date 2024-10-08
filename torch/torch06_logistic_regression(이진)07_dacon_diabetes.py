import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('troch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
#1. 데이터
path = 'C:/AI5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)    #[652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)   # [116 rows x 1 columns]

print(train_csv.shape)  # (652, 9)
print(test_csv.shape)   # (116, 8)
print(submission_csv.shape) # (116, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 노 프라블름

# train_csv = train_csv[train_csv['BloodPressure'] > 0]
# train_csv = train_csv[train_csv['BMI'] > 0.0]

x = train_csv.drop(['Outcome'], axis=1)
print(x)    # [652 rows x 8 columns]
y = train_csv['Outcome']
print(y.shape)  # (652,)

print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=415)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# x_train = torch.DoubleTensor(x_train).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
# y_train = torch.DoubleTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.DoubleTensor(y_test).unsqueeze(1).to(DEVICE)

# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)
# y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)

print("=====================================================")
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(type(x_train), type(y_train))

# ========================================================
# torch.Size([398, 30]) torch.Size([171, 30])
# torch.Size([398, 1]) torch.Size([171, 1])
# <class 'torch.Tensor'> <class 'torch.Tensor'>

# exit()

#2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    # nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    # nn.Linear(32, 32),
    # nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    # nn.ReLU(),
    nn.Sigmoid()

).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()     # 훈련모드, 디폴트
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)

    loss.backward()     # 기울기(gradient)값 계산까지, # 역전파 시작
    optimizer.step()    # 가중치(w) 갱신               # 역전파 끝
    return loss.item()

epochs = 500
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))        #verbose

print("============================================")

#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드 // 역잔파, 가중치 갱신, 기울기 계산할수 있기도 없기도,
                    # 드롭아웃, 배치노멀 <- 얘네들 몽땅 하지마!!!
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", last_loss)

##################### 요 밑에 완성할 것 #############################
from sklearn.metrics import accuracy_score

y_predict = model(x_test)
# print(y_predict)
y_predict = np.round(y_predict.detach().cpu().numpy())
y_test = y_test.cpu().numpy()


accuracy_score = accuracy_score(y_test, y_predict)


# accuracy_score = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))
# print('acc_score :', accuracy_score)
print('acc_score : , {:.4f}'.format(accuracy_score))


# 최종 loss :  26.221479415893555
# acc_score : , 0.7328