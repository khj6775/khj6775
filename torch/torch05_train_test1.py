import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)
# torch :  2.4.1+cu124 사용DEVICE :  cuda

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])
x_predict = np.array([12,13,14])


x = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
print(x.shape, y.shape)

x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)
# exit()

#2. model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 1)
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

for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)

    print('epoch :', epoch, 'loss :', loss)

print('===============================================')
#4. evaluate
def evaluate(model, criterion, x, y):
    model.eval()

    with torch.no_grad():
        y_predict = model(x)

        loss2 = criterion(y, y_predict)

    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)

print('final loss :', loss2)

results = model(torch.Tensor(x_predict).to(DEVICE))

print('12,13,14의 predict :', np.round(results.detach().cpu().numpy(),3))  # numpy 는 cpu만 연산

# final loss : 6.821210263296962e-13
# 12,13,14의 predict : [[12.]
#  [13.]
#  [14.]]