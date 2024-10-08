import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA  = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

print('torch :', torch.__version__, 'device :', DEVICE)

# data
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x = torch.FloatTensor(x)

print(x.shape)  # torch.Size([3])
print(x.size()) # torch.Size([3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3, ) -> (3, 1)

print('before scaling :', x)

x1 = (4 - torch.mean(x)) / torch.std(x)
x = (x - torch.mean(x)) / torch.std(x)

print('after scaling :', x)

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) # (3, ) -> (3, 1)

print(x.size(), y.size())

# model
# model = Sequential()

# model.add(Dense(1, input_dim = 1))

model = nn.Linear(1, 1) # input, output -> y = xw + b

model.to(DEVICE)

# train
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    model.train() # 훈련모드 시작

    optimizer.zero_grad() # 각 배치마다 기울기를 초기화하여 기울기 누적에 의한 문제 해결 (기울기 : loss를 가중치로 미분한 값)

    hypothesis = model(x) # y = wx + b

    loss = criterion(hypothesis, y) # loss = mse -> l = (y - y') ^ 2 / n

    loss.backward() # 기울기(gradient) 값 계산까지 한다 -> 역전파 시작을 선언한다

    optimizer.step() # 가중치(w) 갱신 -> 역전파 끝

    return loss.item()

epochs = 2000

for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)

    print('epoch :', epoch, 'loss :', loss) # verbose

print('=============================================')
# evaluate
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval() # 평가모드

    with torch.no_grad():
        y_predict = model(x)

        loss2 = criterion(y, y_predict)

    return loss2.item()

loss2 = evaluate(model, criterion, x, y)

print('final loss :', loss2)

results = model(torch.Tensor([[(4 - torch.mean(x1)) / torch.std(x)]]).to(DEVICE)).item()

print('4\'s predict :', results)