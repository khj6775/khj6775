import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)  # torch 고정, cpu 고정 
torch.cuda.manual_seed(333) # gpu 고정

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

path = "_data/kaggle/Netflix_stock/netflix-stock-prediction/"
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)    # [967 rows x 6 columns]
print(train_csv.info())
print(train_csv.describe())

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self):
        self.csv = train_csv

        self.x = self.csv.iloc[:, 1:4].values
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        # 정규화

        self.y = self.csv['Close'].values
        self.y = (self.y - np.min(self.y, axis=0)) / (np.max(self.y, axis=0) - np.min(self.y, axis=0))

    def __len__(self):
        return len(self.x) - 30

    def __getitem__(self, i):
    # 시계열 데이터
        x = self.x[i:i+30]
        y = self.y[i+30]

        return x, y

aaa = Custom_Dataset()
print(aaa)

#2. 모델

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTM(input_size=3,
                            hidden_size=32,
                            num_layers=1,
                            batch_first=True
                            )
        self.fc1 = nn.Linear(in_features=30 * 32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.cell(x)
        x = x.contiguous().view(-1, 30 * 32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = LSTM().to(DEVICE)

#3. 컴파일, 훈련
from torch.optim import Adam
# optim = Adam(params=model.parameters(), lr = 0.001)

"""
import tqdm

for epoch in range(1, 201):
    iterator = tqdm.tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()
        
        h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE)  # (num_layers, batch_size, hidden_size) = (5,32,64)
        
        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE), h0)
        
        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))
        
        loss.backward()
        optim.step()
        
        iterator.set_description(f'epoch : {epoch} loss : {loss.item()}')   # iterator print문, 프로그레스바, epo, loss 출력
"""
## save ##
save_path = './_save/torch/'
# torch.save(model.state_dict(), save_path + 't1.pth')

#4. 평가 예측
train_loader = DataLoader(aaa, batch_size=1)

y_predict = []
total_loss = 0
y_true = []

with torch.no_grad():
    model.load_state_dict(torch.load(save_path + 't1.pth', map_location=DEVICE))
    for x_test, y_test in train_loader:

        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE))
        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())

        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)

#print(f'y_predict : {y_predict}, \n shape: {y_predict.shape}')

## 실습 R2 맹글기 ##
from sklearn.metrics import r2_score

y_predict = np.array(y_predict).flatten()
y_true = np.array(y_true).flatten()

r2 = r2_score(y_true, y_predict)
print('R2: ', r2)
print('total_loss: ', total_loss.item())


# R2:  0.97722011278544
# total_loss:  0.0021771183237433434