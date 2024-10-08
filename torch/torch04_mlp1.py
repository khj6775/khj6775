import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],          
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]).transpose()
y = np.array([1,2,3,4,5,6,7,7,9,10])

print(x.shape, y.shape)     # (10, 2) (10,)

### 맹그러봐
# 예측값 : [10, 1.3]

x = torch.FloatTensor(x)

print(x.shape)      # torch.Size([10, 2])
print(x.size())     # torch.Size([10, 2])

x = torch.FloatTensor(x).to(DEVICE)

print('before scaling :', x)

# x1 = (10 - torch.mean(x)) / torch.std(x)
# x = (x - torch.mean(x)) / torch.std(x)

print('after scaling :', x)

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())

#2. model
model = nn.Sequential(
    nn.Linear(2,16),
    nn.Linear(16,32),
    nn.Linear(32,16),
    nn.Linear(16,8),
    nn.Linear(8,4),
    nn.Linear(4,1)
).to(DEVICE)

#3. train
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    
    optimizer.zero_grad()

    hypothesis = model(x)

    loss = criterion(hypothesis, y)

    loss.backward()

    optimizer.step()

    return loss.item()

epochs = 500

for epoch in range(1, epochs +1):
    loss = train(model, criterion, optimizer, x, y)

    print('epoch :', epoch, 'loss :', loss)

print('=============================================')
#4. evaluate
def evaluate(model, criterion, x, y):
    model.eval()

    with torch.no_grad():
        y_predict = model(x)

        loss2 = criterion(y,y_predict)

    return loss2.item()

loss2 = evaluate(model, criterion, x, y)

print('final loss :', loss2)

results = model(torch.Tensor([[10, 1.3]]).to(DEVICE)).item()

print('[10, 1.3] 예측값 : ', results)

# final loss : 0.0802653580904007
# [10, 1.3] 예측값 :  9.83663558959961