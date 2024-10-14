import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('troch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)     # 0,1,2 int 이므로 LongTensor
# print(x.shape, y.shape)     # torch.Size([150, 4]) torch.Size([150])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.75, shuffle=True, random_state=1004,
    stratify=y,                                                    
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # 이거 쓰면 원핫, 소프트맥스 안해도 된다

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 1000   # 특정 고정 상수의 변수는 대문자를 쓴다
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    # print('epoch: {}, loss : {:.8f}'.format(epoch, loss))
    print(f'epoch : {epoch}, loss : {loss:.8f}')
    # 기울기는 계산하고, 가중치는 갱신한다.

#4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()

    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
loss = evaluate(model, criterion, x_test, y_test)
print('loss : ', loss)

############ acc 출력해봐욤 ##############
y_predict = model(x_test)
print(y_predict[:5])
# tensor([[-16.6299,  -2.2799,  19.0287],
#         [-12.7133,   0.7970,  13.6829],
#         [ 23.5782,   7.6652, -38.8507],
#         [ 23.6661,   7.9177, -39.1550],
#         [ -0.1759,  20.1877,  -8.5047]], device='cuda:0',
#        grad_fn=<AddmmBackward0>)
y_predict = torch.argmax(model(x_test), 1)
print(y_predict[:5])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))    # accuracy : 0.9211
print(f'accuracy : {score:.4f}')            # accuracy : 0.9211

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_predict.cpu().numpy())
print('accuracy_score : {:4f}'.format(score2))
print(f'accuracy_score : {score2:4f}')


y_predict = np.round(y_predict.detach().cpu().numpy())
y_test = y_test.cpu().numpy()

accuracy_score = accuracy_score(y_test, y_predict)

# accuracy_score = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))
print('acc_score :', accuracy_score)
# print('acc_score : , {:.4f}'.format(accuracy_score))
