import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10   # vision = image

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

path = './study/torch/_data/'
train_dataset = CIFAR10(path, train=True, download=True)
test_dataset = CIFAR10(path, train=False, download=True)

print(train_dataset)
print(type(train_dataset))  # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])     # (<PIL.Image.Image image mode=L size=28x28 at 0x18172578980>, 5)
print(train_dataset[0][0])

bbb = iter(train_dataset)
# aaa = bbb.next()    # 파이썬 3.9까지 먹히는 문법
aaa = next(bbb)
print(aaa)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets
# print(x_train)
# print(y_train)
print(x_train.shape, len(y_train))    # (50000, 32, 32, 3) 50000

# print(np.min(x_train.numpy()), np.max(x_train.numpy()))     #  0.0 1.0

x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3)      
# reshape와 view = reshape는 아무때나 사용 가능, view는 연속적인 데이터에서만 사용가능 view가 성능이 더 좋다.
print(x_train.shape, len(x_test))   
#(50000, 32, 32, 3) 50000
#(50000, 3072) 10000

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# x_train = torch.DoubleTensor(x_train).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델
class DNN(nn.Module):       # 클래스 괄호 안은 상속
    def __init__(self, num_features):   # 괄호 안 매개변수
        super().__init__()
        # super(self, DNN).__init__

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = DNN(32*32*3).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 0.0001

def train(model, criterion, optimizer, loader):
    # model.train()

    epoch_loss = 0
    epoch_acc = 0
    for x_batch, y_batch in loader:   # (1 epoch = 32*28*28 배치단위)
        
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()         # 배치단위로 기울기 초기화 해야하므로 for문 안에 넣는다.
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        epoch_acc += acc

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, criterion, loader):
    model.eval()

    epoch_loss=0
    epoch_acc=0
    
    with torch.no_grad():
        for x_bacth, y_batch in loader:
            x_bacth, y_batch = x_bacth.to(DEVICE), y_batch.to(DEVICE)

            hypothesis =model(x_bacth)

            loss = criterion(hypothesis, y_batch)

            epoch_loss += loss.item()

            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)
# loss, acc = model.evaluate(x_test, y_test)

epochs = 30
for epoch in range(1, epochs + 1):  # 가독성을 위해 + 1 해준다.
    loss, acc = train(model, criterion, optimizer, train_loader)

    val_loss, val_acc = evaluate(model, criterion, test_loader)

    print('epoch:{}, loss:{:.4f}, acc:{:.3f} val_loss:{:.4f}, val_acc:{:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc
    ))



#4. 평가, 예측
# loss = model.evaluate(x, y)
# def evaluate(model, criterion, loader):
#     model.eval()    # 평가모드 // 역전파, 가중치 갱신, 기울기 계산할수 있기도 없기도,
#                     # 드롭아웃, 배치노멀 <- 얘네들 몽땅 하지마!!!
#     total_loss=0
#     for x_batch, y_batch in loader:
#         with torch.no_grad():
#             y_predict = model(x_batch)
#             loss2 = criterion(y_predict, y_batch)
#             total_loss += loss2.item()
#     return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print("최종 loss : ", last_loss)


from sklearn.metrics import accuracy_score

def acc_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = accuracy_score(y_test, np.round(y_pre.detach().cpu().numpy()).argmax(axis=1))
    return acc

import warnings
warnings.filterwarnings('ignore')

acc = acc_score(model, test_loader)
print('acc_score :', acc)

# 최종 loss :  (1.3976614153423248, 0.5024960063897763)
# acc_score : 0.4825