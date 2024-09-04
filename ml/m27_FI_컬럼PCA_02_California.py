from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = fetch_california_housing(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state
)

model = RandomForestRegressor(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 20)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train1 = []     
for i in rm_index : 
    x_train1.append(x_train[:,i])
x_train1 = np.array(x_train1).T

x_test1 = []     
for i in rm_index : 
    x_test1.append(x_test[:,i])
x_test1 = np.array(x_test1).T

print(x_train1.shape)
print(x_test1.shape)

x_train = np.delete(x_train, rm_index, axis = 1)
x_test = np.delete(x_test, rm_index, axis = 1)


from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_train1 = pca.fit_transform(x_train1)
x_test1 = pca.transform(x_test1)

x_train = np.concatenate([x_train, x_train1], axis=1)
x_test = np.concatenate([x_test, x_test1], axis=1)

model = XGBRegressor(random_state = random_state)
model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

# before remove column
# ------------------- RandomForestRegressor -------------------
# acc : 0.8257590684206899
# [0.51600787 0.05296502 0.04550317 0.03002283 0.03180825 0.13894284
#  0.09250583 0.09224418]


# after remove column (below 20%)
# ------------------- RandomForestRegressor -------------------
# acc : 0.8290357646207668
# [0.52362622 0.05933008 0.05874554 0.14854603 0.10537691 0.10437522]


# PCA 적용 후
# ------------------- XGBRegressor -------------------
# acc : 0.8566939582173253
# [0.4899756  0.07238935 0.04675307 0.14733075 0.10678388 0.10941584
#  0.0273514 ]