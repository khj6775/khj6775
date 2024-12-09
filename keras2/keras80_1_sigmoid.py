import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))     # exp = 지수상수 e=2.xxx

sigmoid = lambda x : 1 / (1 + np.exp(-x))       # lambda = 익명함수

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))

y = sigmoid(x)  

plt.plot(x,y)
plt.grid()
plt.show()