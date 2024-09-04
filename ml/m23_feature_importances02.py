from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (442, 10), (442,)

random_state = 23


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    # stratify=y,
)

#2. 모델구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)

models = [model1, model2, model3, model4]

print('random_state :', random_state)
for model in models:
    model.fit(x_train, y_train)
    print("=============", model.__class__.__name__, "================")   # 클래스 이름만 나오게 한다.
    print('r2', model.score(x_test, y_test))
    print(model.feature_importances_)

# random_state : 11
# ============= DecisionTreeRegressor ================
# r2 0.13904058436276856
# [0.08481726 0.01415506 0.36467009 0.06246466 0.03498836 0.12766765
#  0.06660592 0.00548637 0.16887198 0.07027265]
# ============= RandomForestRegressor ================
# r2 0.5671830693677556
# [0.06320549 0.01447701 0.30768524 0.10444119 0.05373426 0.06862685
#  0.04875988 0.02179593 0.25104785 0.0662263 ]
# ============= GradientBoostingRegressor ================
# r2 0.5339608826988115
# [0.03872999 0.01828807 0.36116286 0.10887614 0.02189599 0.07813943
#  0.04292571 0.01686789 0.26163065 0.05148327]
# ============= XGBRegressor ================
# r2 0.47751299405381575
# [0.0326675  0.09551404 0.2633962  0.08568367 0.05470441 0.08394071
#  0.05644447 0.06577405 0.21835147 0.04352351]