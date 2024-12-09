# 다른말로 swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def mish(x):
#     return x * np.tanh(np.log(1 + np.exp(x)))

selu = lambda x : x * np.tanh(np.log(1 + np.exp(x)))

y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()