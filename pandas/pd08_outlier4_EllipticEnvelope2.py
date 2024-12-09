import numpy as np
aaa = np.array([[-10, 2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

#### for 문 돌려서 맹그러봐 ####

print(aaa.shape)        #(13, 2)

from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.3)   # EllipticEnvelope 을 인스턴스로 객체화 한다.  .3 = 30퍼센트를 이상치로 하겠다.
outliers = EllipticEnvelope()       # default = 0.1, 이상치 10퍼센트

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
