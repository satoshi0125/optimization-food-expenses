import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# データをロード
df_constraint = pd.read_csv('下限値・上限値.csv', index_col=0)
df_data = pd.read_csv('単位あたりの価格・栄養成分.csv', index_col=0)

# スペースがあるとエラーが出るので置換する
df_data.columns = df_data.columns.map(lambda x: x.replace(' ', '-'))
df_constraint.index = df_constraint.index.map(lambda x: x.replace(' ', '-'))

# 最小化対象
objective_name = '価格（税込）'

# 表示
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
display(
    df_data,
    df_constraint.T,
)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')


# 問題宣言
problem = pulp.LpProblem("問題", pulp.LpMinimize)

# 変数宣言
var_ary = [pulp.LpVariable(f'var{i}', 0, sys.maxsize, pulp.LpInteger) for i in range(df_data.shape[0])]

# 目的関数
problem += pulp.lpDot(df_data[objective_name].tolist(), var_ary)

# 制約条件
for constraint_name in df_constraint.index:
    mn, mx = df_constraint.loc[constraint_name, :]
    total = pulp.lpDot(df_data[constraint_name].tolist(), var_ary)
    if mn == mn:
        problem += ( mn <= total , f'{constraint_name}の下限' )
    
    if mx == mx:
        problem += ( total <= mx , f'{constraint_name}の上限' )
    


# ------------------------------------------------------------------------------------------
# 追加で制約
# ------------------------------------------------------------------------------------------

# n-6 / n-3 比
total_n_3 = pulp.lpDot(df_data['脂肪酸_n-3系-多価不飽和_g'].tolist(), var_ary)
total_n_6 = pulp.lpDot(df_data['脂肪酸_n-6系-多価不飽和_g'].tolist(), var_ary)
problem += ( total_n_6 <= 2 * total_n_3 , f'n-6/n-3' )

# ネイチャーメイドは1粒（2*0.5粒）まで
i = df_data['食品情報_食品名'].tolist().index('ネイチャーメイド スーパーマルチビタミン&ミネラル')
problem += ( var_ary[i] <= 2, f'ネイチャーメイド スーパーマルチビタミン&ミネラル は1粒まで' )

# わかめの味噌汁は一杯にする
# ほうれんそうや塩をまとめて食べられて楽なので
# 「カットわかめ」におきかえた場合と価格があまり変わらない
# （わかめの味噌汁の方が数円高い）
i = df_data['食品情報_食品名'].tolist().index('わかめの味噌汁')
problem += ( 1 == var_ary[i], f'わかめの味噌汁 は一杯' )

# オリーブオイルで一価と多価を同量にする
total_1 = pulp.lpDot(df_data['脂肪酸_脂肪酸-多価不飽和_g'].tolist(), var_ary)
total_2 = pulp.lpDot(df_data['脂肪酸_脂肪酸-一価不飽和_g'].tolist(), var_ary)
problem += ( total_1 <= total_2 , f'オリーブオイルで一価と多価を同量にする' )

# ------------------------------------------------------------------------------------------



# 問題表示
print(problem)

# 計算
result_status = problem.solve()

# 結果をまとめる
count = [pulp.value(var) for var in var_ary]

# 結果表示
print('-' * 100)
print('<計算結果>')
print(f"最適性 : {pulp.LpStatus[result_status]}")
print(f"{objective_name} : {pulp.value(problem.objective)}")


'''
個数を df にする
下のセルでも使う
'''

df_count = pd.DataFrame(count, index=df_data['食品情報_食品名'], columns=['個数'])
display(df_count)