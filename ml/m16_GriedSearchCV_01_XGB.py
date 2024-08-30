# 모델 : RandomForestClassifier

prameters =[
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6,10,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist'] },  #12
    {'n_jobs' : [-1,], 'max_depth' : [6,8,10,12],
     'min_samples_leaf' : [3,5,7, 10],  'tree_method' : ['gpu_hist']}, # 16
     {'n_jobs' : [-1,], 'min_samples_leaf' : [3,5,7,10],
     'min_samples_split' : [2,3,5,10], 'tree_method' : ['gpu_hist']}, # 16
     {'n_jobs' : [-1,], 'min_samples_leaf' : [2,3,5,10], 'tree_method' : ['gpu_hist']},   # 4
]   # 48

