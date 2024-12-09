import numpy as np
aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12,50])
print(aaa.shape)        #(13,  )
aaa = aaa.reshape(-1,1)
print(aaa.shape)        #(13, 1)

from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.3)   # EllipticEnvelope 을 인스턴스로 객체화 한다.  .3 = 30퍼센트를 이상치로 하겠다.
outliers = EllipticEnvelope()       # default = 0.1, 이상치 10퍼센트

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
