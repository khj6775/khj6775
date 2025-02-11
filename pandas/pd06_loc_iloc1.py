import pandas as pd
print(pd.__version__)   #1.5.3

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]

index = ['031', '059', '033', '045','023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)

print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000     
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
print("========================================================")
# print(df[0])    # error
# print(df['031'])    # error
print(df['시가'])   # "판다스열행" 이기 때문에  """ 컬럼이 기준 """"

#### 아모레 출력하고 싶어
# print(df[3, 0])    # keyerror
# print(df['045', '종목명'])  # keyerror
# print(df['종목명', '045'])  # keyerror
print(df['종목명']['045'])  # 아모레     # 판다스 열행이라 요건 나오네
print("========================================================")

# loc : 인덱스를 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출
        # 인트loc = 인트 로케이션!!!! 이렇게 외워라!!!!
################################################################
print("====================== 아모레 뽑자 ===========================")
# print(df.iloc['045'])   # 에러
print(df.iloc[3])

# print(df.loc[3])  # 에러
print(df.loc['045'])

print("====================== 네이버 뽑자 ===========================")
print(df.loc['023'])
print(df.iloc[4])

print("====================== 아모레 종가 뽑자 ===========================")
print(df.loc['045']['종가'])        # 6000
print(df.loc['045','종가'])         # 6000
print(df.loc['045'].loc['종가'])    # 6000

print(df.iloc[3][2])    # 6000  판다스1에서는 되고 판다스2에서는 워닝뜨지만 실행된다.
print(df.iloc[3, 2])    # 6000  
print(df.iloc[3].iloc[2])    # 6000  

print(df.loc['045'][2])     # 6000
# print(df.loc['045', 2])     # error

print(df.iloc[3]['종가'])     # 6000
# print(df.iloc[3, '종가'])   # error

print(df.loc['045'].iloc[2])    # 6000
print(df.iloc[3].loc['종가'])   # 6000



# loc 는 판다스 열과행의 인덱스를 직접 넣어서 출력
# iloc 는 판다스 열과행의 인덱스 위치값(숫자)을 넣어서 출력 

