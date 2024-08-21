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

scaler = StandardScaler()
x = scaler.fit_transform(x)

for i in range(4, 0, -1):
 pca = PCA(n_components=i)
x = pca.fit_transform(x)


# pca = PCA(n_components=1)   # 4개가 3개가 된다.
# x = pca.fit_transform(x)

# 통상적으로 스켈일링을 먼저하고 PCA를 한다.

print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=7, shuffle=True,
    stratify=y,  # y 를 기준으로 같은 값으로 분류, 분류에서만 쓴다.
)




#2. 모델
model = RandomForestClassifier(random_state=7)    
# 트리구조, 디시젼트리 찾아보기, 디시젼트리의 앙상블 모델이다. 디폴트85점 젤좋다. 3대장은 80점정도, 다음에 배움.

#3. 훈련.
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print(x.shape)
print('model.score : ', results)

# (150, 3)
# model.score :  0.9   = accuracy score
# (150, 4)
# model.score :  0.9333333333333333
# (150, 2)
# model.score :  0.9333333333333333
# (150, 1)
# model.score :  0.9333333333333333

# (150, 4)
# model.score :  1.0
# (150, 3)
# model.score :  1.0
# (150, 2)
# model.score :  0.9333333333333333
# (150, 1)
# model.score :  0.9333333333333333