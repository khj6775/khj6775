from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier   # 분류면 Classifier, 회귀면 Regressor
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=7, shuffle=True,
    stratify=y,  # y 를 기준으로 같은 값으로 분류, 분류에서만 쓴다.
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3)   # 4개가 3개가 된다.
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#2. 모델
model = RandomForestClassifier(random_state=7)    
# 트리구조, 디시젼트리 찾아보기, 디시젼트리의 앙상블 모델이다. 디폴트85점 젤좋다. 3대장은 80점정도, 다음에 배움.

#3. 훈련.
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print(x.shape)
print('model.score : ', results)

# (150, 4)
# model.score :  1.0
# (150, 3)
# model.score :  1.0
# (150, 2)
# model.score :  0.9333333333333333
# (150, 1)
# model.score :  1.0