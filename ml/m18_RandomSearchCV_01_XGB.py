prameters =[
    {'n_jobs' : [-1,], 'n_estimators' : [100, 250, 500], 'max_depth' : [4,6,8,10,12],
     'min_samples_leaf' : [3,5, 10], 'tree_method' : ['gpu_hist'], 'learnig rate' : 0.002 },  #12
    {'n_jobs' : [-1,], 'max_depth' : [6,8,10,12], 'learnig rate' : 0.003,
     'min_samples_leaf' : [3,5,7,9, 10],  'tree_method' : ['gpu_hist']}, # 16
     {'n_jobs' : [-1,], 'min_samples_leaf' : [3,5,7,10], 'learnig rate' : 0.004,
     'min_samples_split' : [2,3,5,8,10], 'tree_method' : ['gpu_hist']}, # 16
     {'n_jobs' : [-1,], 'min_samples_leaf' : [2,3,5,8,10], 'tree_method' : ['gpu_hist'], 'learnig rate' : 0.005},   # 4
]   # 48

### 경우의수를 100개로 늘려서 랜덤서치!!!
# 러닝레이트 반드시 넣고
# 다른 파라미터도 둬개 더 넣어라

parameters = [
    {"C":[1, 10, 100, 1000], 'kernel' :['linear', 'sigmoid'], 'degree':[3,4,5]},  # 24개 파라미터
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},   # 6개 파라미터
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'],
     'gamma':[0.01, 0.001, 0.0001], 'degree':[3,4]}     # 24
]   # 54