import pandas as pd
import datetime
import numpy as np
import sys, os
loan_11_dif = pd.read_csv('../feature_extraction/midProcessed/loan_merge_11_diff.csv').drop(['Unnamed: 0'],axis=1)
loan_11 = pd.read_csv('../feature_extraction/midProcessed/t_loan_11.csv').drop(['Unnamed: 0'],axis=1)
loan_11_source = pd.read_csv('../data/t_loan_sum.csv')
loan_11_source.drop('month',axis=1,inplace=True)
loan_11_source.rename(columns={'loan_sum':'label'},inplace=True)
loan_11.drop(['loan_time','one_plan_loan'],axis=1,inplace=True)
loan_11['exp_loan_amount'] = np.exp(loan_11['loan_amount'])
'''
,uid,loan_time,loan_amount,plannum,one_plan_loan

'''
def f(x):
    return [list(tup) for tup in zip(x['loan_amount'], x['plannum'], x['exp_loan_amount'])]
loan_11_agg = loan_11.groupby('uid').apply(f)
loan_11_agg = pd.DataFrame(loan_11_agg.values.tolist(),index=loan_11_agg.index)
loan_11_out = pd.DataFrame(index=loan_11_agg.index)
for col in  loan_11_agg.columns:
    colist = loan_11_agg[col].apply(pd.Series)
    loan_11_out = pd.concat([loan_11_out,colist],axis=1)
loan_11_out.fillna(0,inplace=True)
loan_11_out['uid'] = loan_11_out.index
loan_11_out = pd.merge(loan_11_out,loan_11_source,how='left',on='uid')
print(loan_11_out)