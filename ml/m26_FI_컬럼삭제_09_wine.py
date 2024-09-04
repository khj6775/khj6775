from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = load_wine(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state,
    stratify=y
)

model = DecisionTreeClassifier(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 10)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

x_train = np.delete(x_train, rm_index, axis = 1)
x_test = np.delete(x_test, rm_index, axis = 1)

model = DecisionTreeClassifier(random_state = random_state)
model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

# before remove column
# ------------------- DecisionTreeClassifier -------------------
# acc : 0.8333333333333334
# [0.         0.00674162 0.         0.02067613 0.02095894 0.
#  0.39314913 0.01427638 0.         0.02855276 0.         0.11353687
#  0.40210817]


# after remove column (below 10%)
# ------------------- DecisionTreeClassifier -------------------
# acc : 0.8888888888888888
# [0.00674162 0.03495251 0.02095894 0.39314913 0.         0.02855276
#  0.11353687 0.40210817]