from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = load_diabetes(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state
)

model = RandomForestRegressor(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 10)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

x_train = np.delete(x_train, rm_index, axis = 1)
x_test = np.delete(x_test, rm_index, axis = 1)

model = RandomForestRegressor(random_state = random_state)
model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

# before remove column
# ------------------- RandomForestRegressor -------------------
# acc : 0.5346652880806231
# [0.0613852  0.01352996 0.29480687 0.11281554 0.04864204 0.04697832
#  0.05707059 0.02583196 0.2708351  0.06810441]


# after remove column (below 10%)
# ------------------- RandomForestRegressor -------------------
# acc : 0.5392037190625173
# [0.06192376 0.29479594 0.11450466 0.05148664 0.04934926 0.0570283
#  0.02678765 0.27237525 0.07174854]