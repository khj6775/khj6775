from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = load_digits(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state,
    stratify=y
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
# acc : 0.9583333333333334
# [0.04149784 0.00855798 0.00640059 0.0049863  0.04138929 0.0121838
#  0.00816269 0.01016056 0.01153683 0.0065804  0.02144895 0.00968225
#  0.00537887 0.00425924 0.00429313 0.0388104  0.00934291 0.045308
#  0.00405606 0.00655915 0.03591139 0.01019387 0.02878592 0.02226923
#  0.01899081 0.06414141 0.00795486 0.01344679 0.05312642 0.01392515
#  0.04224245 0.01242405 0.03182242 0.04346353 0.01092045 0.01155525
#  0.02489754 0.00342907 0.00479034 0.00826202 0.00715778 0.00851775
#  0.01459819 0.02161208 0.00284656 0.01333367 0.00248005 0.06523447
#  0.01250932 0.02016074 0.04075119 0.02764995]