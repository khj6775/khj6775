param_bounds= {'x1' : (9, 15),      # 0
               'x2' : (3, 6)}       # 2

def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 +10

# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,
    pbounds=param_bounds,
    random_state=333,
)

optimizer.maximize(init_points=5, 
                   n_iter=20,)
print(optimizer.max)

# 하이퍼 파라미터 튜닝을 잘 잡아준다. 짱조음.