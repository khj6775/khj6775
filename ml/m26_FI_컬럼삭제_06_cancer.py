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
    stratify=y
)

model = RandomForestClassifier(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 10)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

x_train = np.delete(x_train, rm_index, axis = 1)
x_test = np.delete(x_test, rm_index, axis = 1)

model = RandomForestClassifier(random_state = random_state)
model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

# before remove column
# ------------------- RandomForestRegressor -------------------
# acc : 0.9210526315789473
# [0.06272787 0.00997879 0.04507715 0.02529859 0.00576127 0.00952573
#  0.05451327 0.12860048 0.00250562 0.00326888 0.01580793 0.00405118
#  0.01076959 0.02974786 0.00467666 0.00421555 0.00865983 0.00459093
#  0.0032947  0.00562627 0.12945362 0.01669315 0.14227724 0.0609324
#  0.01806046 0.02449389 0.0334624  0.12353544 0.0062715  0.00612176]


# after remove column (below 10%)
# ------------------- RandomForestRegressor -------------------
# acc : 0.956140350877193
# [0.04203301 0.01104091 0.0322968  0.02793093 0.00444946 0.02185808
#  0.04221508 0.11613149 0.00496342 0.00573707 0.0067839  0.02169112
#  0.00527581 0.00526966 0.00547818 0.00741168 0.00580847 0.13000433
#  0.0139548  0.14946318 0.12493668 0.01451341 0.02641819 0.03856364
#  0.1190102  0.00852078 0.00823973]