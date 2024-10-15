import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('troch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
path = 'C:/AI5/_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'replaced_train.csv', index_col=0)
print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'replaced_test.csv', index_col=0)
print(test_csv)     # [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [110023 rows x 1 columns]

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(submission_csv.shape) # (110023, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 결측치 없음

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
train_csv=train_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis = 1).values
print(x)    # [165034 rows x 10 columns]

y = train_csv['Exited'].values
print(y.shape)      # (165034,)


print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=369,
    stratify=y,
    )
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module):                         # 클래스 정의
    def __init__(self, input_dim, output_dim):  # input_dim, output_dim 받아들이는 인자
        # super().__init__()  # 디폴트 # nn.Module 에 있는걸 다 쓰겠다
        super(Model, self).__init__()   # 아빠
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,32)
        self.linear5 = nn.Linear(16,output_dim)
        self.linear4 = nn.Linear(32,16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    # 순전파 !!!
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x
    
# 클래스를 인스턴스 화 한다.
model = Model(10, 1).to(DEVICE)

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

epochs = 200
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
        loss2 = criterion(y_predict, y)
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
print('acc_score :', accuracy_score)
print('acc_score : {:.4f}'.format(accuracy_score))

# 최종 loss :  19.32361602783203
# acc_score : 0.8641514006988346