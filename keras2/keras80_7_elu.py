# 다른말로 swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def elu(x, a=1.0):
#     return np.where(x>0, x, a*(np.exp(x)-1))

# elu = lambda x, a = 1.0 : np.where(x>0, x, a*(np.exp(x)-1))
selu = lambda x: np.where(x > 0, 1.0507 * x, 1.0507 * 1.67326 * (np.exp(x) - 1))

y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()