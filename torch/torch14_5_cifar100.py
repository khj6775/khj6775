import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])


path = './_data/torch_data/'
train_dataset = CIFAR100(path, train=True, download=True, transform=transf)
test_dataset = CIFAR100(path, train=True, download=True, transform=transf)

print(train_dataset[0][0])  # torch.Size([1, 150, 150])   채널이 맨앞으로 간다.
print(train_dataset[0][1])  # 5
print(train_dataset[0][0])

print(train_dataset)

# x_train, y_train = train_dataset.data/255. , train_dataset.targets
# x_test, y_test = test_dataset.data/255. , test_dataset.targets

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

bbb = iter(train_loader)
aaa = next(bbb)
print(aaa[0].shape)
print(len(train_loader))

#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5),
        )
        self.hidden_layer4 = nn.Linear(16*5*5, 16)
        self.output_layer = nn.Linear(in_features=16, out_features=100)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)   # x.shape[0]  x 의 배치
        # x = flatten()(x) # 케라스, 윗줄과 같은 뜻
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = CNN(3).to(DEVICE)   # 채널만 넣으면 알아서 맞춰준다.

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

epochs = 10
for epoch in range(1, epochs + 1):  # 가독성을 위해 + 1 해준다.
    loss, acc = train(model, criterion, optimizer, train_loader)

    val_loss, val_acc = evaluate(model, criterion, test_loader)

    print('epoch:{}, loss:{:.4f}, acc:{:.3f} val_loss:{:.4f}, val_acc:{:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc
    ))

last_loss = evaluate(model, criterion, test_loader)
print("최종 loss : ", last_loss)


# from sklearn.metrics import accuracy_score

# def acc_score(model, loader):
#     x_test = []
#     y_test = []
#     for x_batch, y_batch in loader:
#         x_test.extend(x_batch.detach().cpu().numpy())
#         y_test.extend(y_batch.detach().cpu().numpy())
#     x_test = torch.FloatTensor(x_test).to(DEVICE)
#     y_pre = model(x_test)
#     acc = accuracy_score(y_test, np.round(y_pre.detach().cpu().numpy()).argmax(axis=1))
#     return acc

# import warnings
# warnings.filterwarnings('ignore')

# acc = acc_score(model, test_loader)
print('acc_score :', acc)

# 최종 loss :  (3.4217357614142574, 0.18336132437619962)
# acc_score : 0.17810300703774792