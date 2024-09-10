from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=334, train_size=0.8,
    # stratify=y,
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {                                      # 여기1
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 5,
}

#2. 모델
model = XGBRegressor(random_state=334, **parameters)    #여기 2

model.set_params(gamma=0.4, learning_rate=0.2)     # 여기3  파라미터를 여기 1, 2 ,3에 다 넣을 수 있다.

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("사용파라미터 : ", model.get_params())
results = model.score(x_test, y_test)
print('최종점수 :', results)
