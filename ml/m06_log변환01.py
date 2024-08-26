import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)
print(data)
print(data.shape)   # (1000,)
print(np.min(data), np.max(data))
# 0.0001981593758429286 17.23567303890776

log_data = np.log1p(data)       # log0 이 없기 때문에 log1p 를 사용

exp = np.exp(log_data)

# 원본 데이터 히스토그램 그리자
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')
# plt.show()

# 로그변환 데이터 히스토그램 그리자
plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
# plt.show()

# print(log_data)
# print(exp)