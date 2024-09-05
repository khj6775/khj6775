import pandas as pd

df = pd.DataFrame({'A' : [1, 2, 3, 4, 5],
                   'B' : [10,20,30,40,50],
                   'C' : [5, 4, 3, 2, 1],
                   'D' : [3, 7, 5, 1, 4],
                   })
correlations = df.corr()
print(correlations)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)          #0.11.2 
print(matplotlib.__version__)   #3.4.3
# sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,         # 표안에 수치 명시
            cbar=True)          # 사이드 바
plt.show()
