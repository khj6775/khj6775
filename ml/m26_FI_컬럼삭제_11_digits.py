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

model = DecisionTreeClassifier(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 40)

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
# acc : 0.8611111111111112
# [0.         0.00175266 0.0059297  0.         0.01128237 0.04645375
#  0.013399   0.         0.         0.02477994 0.02107961 0.00206196
#  0.04438082 0.0314279  0.00140588 0.         0.00077323 0.00629771
#  0.00579926 0.01940361 0.00734166 0.07810215 0.         0.
#  0.         0.00074231 0.06386257 0.05216682 0.00657471 0.0049206
#  0.01526654 0.         0.         0.10314942 0.00548628 0.00292884
#  0.07397949 0.02074196 0.00446831 0.         0.         0.00422266
#  0.07475178 0.05610698 0.0100389  0.00077323 0.00571869 0.
#  0.         0.         0.01464198 0.00752311 0.00590972 0.00615519
#  0.02212545 0.00077323 0.         0.         0.01313429 0.
#  0.06663634 0.02965036 0.0014176  0.00446141]


# after remove column (below 40%)
# acc : 0.8638888888888889
# [0.02766788 0.00752976 0.04844307 0.013399   0.01390387 0.02089714
#  0.04557074 0.02558788 0.00773125 0.00777139 0.01825367 0.00811489
#  0.07810215 0.06806203 0.05361019 0.0079936  0.00373101 0.01707075
#  0.10392266 0.0082364  0.00781428 0.07397949 0.0176647  0.01045085
#  0.00281678 0.07546155 0.05319647 0.00885194 0.00607954 0.00994634
#  0.00742002 0.00414969 0.00969902 0.02049814 0.00745619 0.07003857
#  0.02887713 0.        ]