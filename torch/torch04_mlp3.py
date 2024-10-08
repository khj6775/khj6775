import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],          
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10,9,8,7,6,5,4,3,2,1]
              ]).T
print(x.shape, y.shape)     # (10, 3) (10,)

### 맹그러봐
# 예측값 : [10, 31, 211]
x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
print(x.shape, y.shape)

# exit()

#2. model
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 3)
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

loss2 = evaluate(model, criterion, x, y)

print('final loss :', loss2)

results = model(torch.Tensor([[10, 31, 211]]).to(DEVICE))

print(results)      # tensor([[ 1.1000e+01,  1.5733e+00, -1.9372e-06]], device='cuda:0',
                    # grad_fn=<AddmmBackward0>) 

print(results.detach())  # tensor([[ 1.1000e+01,  1.5733e+00, -2.7642e-05]], device='cuda:0')

# print(results.detach().numpy()) # numpy는 cpu에서만 돌아서 에러

print(results.detach().cpu().numpy())

print('10, 31, 211 의 predict :', np.round(results.detach().cpu().numpy(),3))  # numpy 는 cpu만 연산

# final loss : 0.005317135713994503
# 10, 31, 211 의 predict : [[11.     1.573 -0.   ]]

# Torch 는 행렬연산, 그래프 연산, 그래프연산을 없애주는게 detach()