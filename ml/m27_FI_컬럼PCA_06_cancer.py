from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = load_breast_cancer(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state,
    stratify=y,
)

model = RandomForestClassifier(random_state = random_state)

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

model = XGBClassifier(random_state = random_state)
model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)


# PCA 적용 후
# ------------------- XGBClassifier -------------------
# acc : 0.9385964912280702
# [0.01781359 0.01163854 0.00059943 0.         0.00801999 0.00363082
#  0.0144502  0.5324745  0.0058312  0.0029903  0.00637623 0.0028827
#  0.00189913 0.00330867 0.06645722 0.01439862 0.06252789 0.02083507
#  0.00954518 0.00522338 0.02536897 0.16569892 0.00643484 0.00844667
#  0.003148  ]