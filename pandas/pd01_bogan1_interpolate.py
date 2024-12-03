"""
결측치 처리
1. 삭제 - 행 또는 열 - 데이터가 많을 때
2. 임의의 값
 평균 : mean
 중위 : median
 0 : fillna
 앞값 : ffill
 뒷값 : bfill
 특정값 : 777 (먼가 조건을 같이 넣는게 좋것지?)
 기타등등
3. interpolate 보간법
4. 모델 : .predict     다른 모델 쓰기.
5. 부스팅 계열 모델 : 이상치와 결측치에 자유롭다.
"""

import pandas as pd
import numpy as np

dates = ['10/11/2024', '10/12/2024', '10/13/2024',
         '10/14/2024', '10/15/2024', '10/16/2024']
dates = pd.to_datetime(dates)
print(dates)

print("==================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index = dates)
print(ts)

ts = ts.interpolate()      # 끝에 값은 마지막 값과 같아지는 것 주의
print(ts)