import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('troch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=369,
    # stratify=y,
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





# DataLoader
# x, y 를 합친다. 배치를 정해준다. 끝.
from torch.utils.data import TensorDataset  # x,y 합친다.
from torch.utils.data import DataLoader     # batch 정의.

# 토치데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x0000012A61278E20>
print(type(train_set))  # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set))   # 398
print(train_set[0])     # 튜플로 되어있다.
print(train_set[0][0])     # 첫번째 x
print(train_set[0][1])     # 첫번째 y   train_set[397] 까지 있다

# 토치데이터셋 만들기 2. batch를 너준다. 끝!!!
train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
print(len(train_loader))    # 10    398/40
print(train_loader)     # <torch.utils.data.dataloader.DataLoader object at 0x000001C3A765E6A0>
# print(train_loader[0])  # 이터레이터라서 리스트를 볼때처럼 보려고 하면 에러가 생긴다.

print("==================================================================")
#1. 이터레이터를 for문으로 확인.
# for aaa in train_loader:
#     print(aaa)
#     break

bbb = iter(train_loader)
# aaa = bbb.next()    # 파이썬 3.9까지 먹히는 문법
aaa = next(bbb)
print(aaa)
print(type(aaa))    # <class 'list'>
print(len(aaa))     # 2
print(aaa[0][1])






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
        # x = self.sigmoid(x)
        return x
    
# 클래스를 인스턴스 화 한다.
model = Model(10, 1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()     # 훈련모드, 디폴트
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()           # 전부 배치 단위로 돌린다
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)

        loss.backward()     # 기울기(gradient)값 계산까지, # 역전파 시작
        optimizer.step()    # 가중치(w) 갱신               # 역전파 끝
        total_loss += loss.item()
    return total_loss / len(loader)  # 배치 단위 10개의 loss 가 나오므로 평균을 구한다.

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {}'.format(epoch, loss))        #verbose

print("============================================")

#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, loader):
    model.eval()    # 평가모드 // 역잔파, 가중치 갱신, 기울기 계산할수 있기도 없기도,
                    # 드롭아웃, 배치노멀 <- 얘네들 몽땅 하지마!!!
    total_loss=0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch, y_predict)
            total_loss += loss2.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print("최종 loss : ", last_loss)

##################### 요 밑에 완성할 것 (데이터 로더를 사용하는것으로 바꿔라) #############################
from sklearn.metrics import accuracy_score, r2_score

# y_predict = model(x_test)
# # print(y_predict)
# y_predict = y_predict.detach().cpu().numpy()
# y_test = y_test.cpu().numpy()

# r2_score = r2_score(y_test, y_predict)

# accuracy_score = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))
# print('r2_score :', r2_score)
# print('r2_score : {:.4f}'.format(r2_score))

def R2_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = r2_score(y_test, y_pre.detach().cpu().numpy())
    return acc

import warnings
warnings.filterwarnings('ignore')

acc = R2_score(model, test_loader)
print('r2_score :', acc)

# 최종 loss :  2781.40625
# r2_score : 0.5504147410392761