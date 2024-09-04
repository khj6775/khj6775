# california, diabetes
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
dataset1 = fetch_california_housing()
dataset2 = load_diabetes()

datasets = [dataset1, dataset2]
name = ['fetch_california_housing', 'load_diabetes']



i=0
for dataset in datasets:
    x = dataset.data
    y = dataset.target

    random_state = 11

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
              x, y, train_size=0.8, random_state=random_state,
            #   stratify=y,
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("=============", name[i], "================")   # 클래스 이름만 나오게 한다.

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
        print('acc', model.score(x_test, y_test))
        print(model.feature_importances_)
    i=i+1


# ============= fetch_california_housing ================
# random_state : 11
# ============= DecisionTreeRegressor ================
# acc 0.6277782516666432
# [0.51028316 0.05047653 0.03270094 0.02410103 0.0291914  0.13066737
#  0.10633285 0.11624673]
# ============= RandomForestRegressor ================
# acc 0.8128382274570127
# [0.52270801 0.05166692 0.04587657 0.02917759 0.03116869 0.13627083
#  0.09137071 0.09176069]
# ============= GradientBoostingRegressor ================
# acc 0.7892500844269648
# [0.59894375 0.03140002 0.02279154 0.00459228 0.00330813 0.12521347
#  0.10686456 0.10688626]
# ============= XGBRegressor ================
# acc 0.8386771412035462
# [0.4808104  0.06665614 0.04903776 0.02331535 0.02384444 0.14351654
#  0.10850842 0.10431097]
# ============= load_diabetes ================
# random_state : 11
# ============= DecisionTreeRegressor ================
# acc 0.13904058436276856
# [0.08481726 0.01415506 0.36467009 0.06246466 0.03498836 0.12766765
#  0.06660592 0.00548637 0.16887198 0.07027265]
# ============= RandomForestRegressor ================
# acc 0.5679629495076811
# [0.06320549 0.01447701 0.30768524 0.10444119 0.05373426 0.06862685
#  0.04875988 0.02179593 0.25104785 0.0662263 ]
# ============= GradientBoostingRegressor ================
# acc 0.5329598130879751
# [0.03872999 0.01828807 0.36116286 0.10887614 0.02189599 0.07813943
#  0.04292571 0.01686789 0.26163065 0.05148327]
# ============= XGBRegressor ================
# acc 0.47751299405381575
# [0.0326675  0.09551404 0.2633962  0.08568367 0.05470441 0.08394071
#  0.05644447 0.06577405 0.21835147 0.04352351]