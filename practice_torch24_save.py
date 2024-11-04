import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

path = '_data/kaggle/Netflix_stock/netflix-stock-prediction/'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)
print(train_csv.info())
print(train_csv.describe())

#import matplotlib.pyplot as plt
#data = train_csv.iloc[:, 1:4]
#data['종가'] = train_csv['Close']
#print(data)

#hist = data.hist()
#plt.show()

# 컬럼 별이 아닌 전체의 최대 최소 기준으로 계산되는 문제 해결하기
#data = train_csv.iloc[:, 1:4].values
#data = (data - np.min(data)) / (np.max(data) - np.min(data))
#data = pd.DataFrame(data)
#print(data.describe())

#                0           1           2
#count  967.000000  967.000000  967.000000
#mean     0.419602    0.429021    0.409107
#std      0.304534    0.309121    0.298979
#min      0.002915    0.014577    0.000000
#25%      0.128280    0.134111    0.125364
#50%      0.332362    0.338192    0.326531
#75%      0.725948    0.734694    0.708455
#max      0.994169    1.000000    0.970845

# axis = 0로 열 기준으로 
#data = train_csv.iloc[:, 1:4].values
#data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
#data = pd.DataFrame(data)
#print(data.describe())
#                 0           1           2
# count  967.000000  967.000000  967.000000
# mean     0.420363    0.420574    0.421392
# std      0.307221    0.313694    0.307957
# min      0.000000    0.000000    0.000000
# 25%      0.126471    0.121302    0.129129
# 50%      0.332353    0.328402    0.336336
# 75%      0.729412    0.730769    0.729730
# max      1.000000    1.000000    1.000000

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
        x = self.x[ i:i+30]
        y = self.y[i+30]

        return x, y

aaa = Custom_Dataset()

train_loader = DataLoader(aaa, batch_size=32)

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
optim = Adam(params=model.parameters(), lr = 0.001)

import tqdm

for epoch in range(1, 201):
    iterator = tqdm.tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()

        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE))

        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))

        loss.backward()
        optim.step()

        iterator.set_description(f'epoch : {epoch} loss : {loss.item()}')

## save ##
save_path = './_save/torch/'
torch.save(model.state_dict(), save_path + 't1.pth')