import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST   # vision = image

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])    # 앞의 (0.5)=평균, 뒤의 (0.5)=표편

# minmax(x_train) - 평균(0.5)# 고정
# --------------------------------- = Z_Score Normalization (정규화와 표준화의 짬뽕)
#          표편(0.5)# 고정



#1. 데이터
path = './_data/torch_data/'
# train_dataset = MNIST(path, train=True, download=False, )
# test_dataset = MNIST(path, train=False, download=False, )

train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

print(train_dataset[0][0].shape)  # torch.Size([1, 150, 150])   채널이 맨앞으로 간다.
print(train_dataset[0][1])  # 5
print(train_dataset[0][0])

##### 정규화(MinMax)  /255.
# x_train, y_train = train_dataset.data/255. , train_dataset.targets   # 이거하면 shape 28 로 롤백.
# x_test, y_test = test_dataset.data/255. , test_dataset.targets
# print(x_train.shape, y_train,size())

### x_train/127.5 - 1 # 얘의 값의 범위는?   -1 ~ 1  # 정규화라기보다는 표준화와 가깝다. (Z Score-정규화)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

##################### 잘 받아졌는지 확인한거여 ##########################
bbb = iter(train_loader)
# aaa = bbb.next()
aaa = next(bbb)
# print(aaa)
print(aaa[0].shape) # torch.Size([32, 1, 56, 56])   채널은 컬러
print(len(train_loader))    # 1875 = 60000 / 32

#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),   #(n, 64, 54, 54)
            # model.Conv2D(64, (3,3), stride=1, input_shape=(56, 56, 1))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),        # (n, 64, 27, 27)
            nn.Dropout(0.5),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), # (n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),         # (n, 32, 12, 12)
            nn.Dropout(0.5),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), # (n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),         # (n, 16, 5, 5)
            nn.Dropout(0.5),
        )

        self.hidden_layer4 = nn.Linear(16*5*5, 16)
        self.output_layer = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)   # x.shape[0]  x 의 배치
        # x = flatten()(x) # 케라스, 윗줄과 같은 뜻
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = CNN(1).to(DEVICE)   # 채널만 넣으면 알아서 맞춰준다.

# model.summary()     # 텐서플로
print(model)    # 깔끔하지 않다.

from torchsummary import summary    # 모델 요약
summary(model, (1, 56, 56))     

# 토치는 채널이 앞에 온다.     채널, 가로, 세로 순

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 54, 54]             640
#               ReLU-2           [-1, 64, 54, 54]               0
#          MaxPool2d-3           [-1, 64, 27, 27]               0
#            Dropout-4           [-1, 64, 27, 27]               0
#             Conv2d-5           [-1, 32, 25, 25]          18,464
#               ReLU-6           [-1, 32, 25, 25]               0
#          MaxPool2d-7           [-1, 32, 12, 12]               0
#            Dropout-8           [-1, 32, 12, 12]               0
#             Conv2d-9           [-1, 16, 10, 10]           4,624
#              ReLU-10           [-1, 16, 10, 10]               0
#         MaxPool2d-11             [-1, 16, 5, 5]               0
#           Dropout-12             [-1, 16, 5, 5]               0
#            Linear-13                   [-1, 16]           6,416
#            Linear-14                   [-1, 10]             170
# ================================================================
# Total params: 30,314
# Trainable params: 30,314
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 3.97
# Params size (MB): 0.12
# Estimated Total Size (MB): 4.09
# ----------------------------------------------------------------





'''
#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 0.0001

def train(model, criterion, optimizer, loader):

    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()

        hypothesis = model(x_batch)

        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        epoch_acc += acc.item()

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

# 최종 loss :  (0.041599708732016086, 0.989117412140575)
# acc_score : 0.9884
'''