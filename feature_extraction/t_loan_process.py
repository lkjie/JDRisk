import pandas as pd
import datetime
import numpy as np

# t_click = pd.read_csv('../data/t_click.csv')
t_loan = pd.read_csv('../data/t_loan.csv')
# t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
# t_order = pd.read_csv('../data/t_order.csv')
t_user = pd.read_csv('../data/t_user.csv')

t_loan['exp_loan_amount'] = np.exp(t_loan['loan_amount'])
# loan
# 每月借款总额，每周借款总额，每月借款次数，每周借款次数，4个月借款总额，8-11月每月平均借款，8-10月每月平均借款（排除双十一），借款总期数，每次借款平均期数，每月借款平均期数
'''
2016-08-03(周三)到2016-11-30(周三)
'''
t_loan_8 = t_loan[(t_loan['loan_time'] >= "2016-08-01 00:00:00") & (t_loan['loan_time'] < "2016-09-01 00:00:00")]
t_loan_8_sum = t_loan_8.groupby('uid').sum()
t_loan_8_sum.rename(columns=lambda x: '8_' + x, inplace=True)
t_loan_8_sum.reset_index(level=['uid'], inplace=True)
t_loan_9 = t_loan[(t_loan['loan_time'] >= "2016-09-01 00:00:00") & (t_loan['loan_time'] < "2016-10-01 00:00:00")]
t_loan_9_sum = t_loan_9.groupby('uid').sum()
t_loan_9_sum.rename(columns=lambda x: '9_' + x, inplace=True)
t_loan_9_sum.reset_index(level=['uid'], inplace=True)
t_loan_10 = t_loan[(t_loan['loan_time'] >= "2016-10-01 00:00:00") & (t_loan['loan_time'] < "2016-11-01 00:00:00")]
t_loan_10_sum = t_loan_10.groupby('uid').sum()
t_loan_10_sum.rename(columns=lambda x: '10_' + x, inplace=True)
t_loan_10_sum.reset_index(level=['uid'], inplace=True)
t_loan_11 = t_loan[(t_loan['loan_time'] >= "2016-11-01 00:00:00") & (t_loan['loan_time'] < "2016-12-01 00:00:00")]
t_loan_11_sum = t_loan_11.groupby('uid').sum()
t_loan_11_sum.rename(columns=lambda x: '11_' + x, inplace=True)
t_loan_11_sum.reset_index(level=['uid'], inplace=True)


# 分析loan和loan_sum的关系
t_loan_11 = t_loan[(t_loan['loan_time'] >= "2016-11-01 00:00:00") & (t_loan['loan_time'] < "2016-12-01 00:00:00")]
t_loan_11['one_plan_exp_loan'] = t_loan_11['exp_loan_amount'] / t_loan_11['plannum']
t_loan_11.groupby('uid').count()
t_loan_11_sum = t_loan_11.groupby('uid').sum()
t_loan_11_sum.rename(columns=lambda x: '11_' + x, inplace=True)
t_loan_11_sum.reset_index(level=['uid'], inplace=True)
t_loan_11_sum.to_csv('trainProcessed/t_loan_11_sum.csv')

t_loan_sum = pd.read_csv('trainProcessed/exp_t_loan_sum.csv')
t_loan_11 = pd.read_csv('trainProcessed/t_loan_11_sum.csv')
merge_11 = pd.merge(t_loan_sum,t_loan_11,on='uid')
merge_11.drop(['Unnamed: 0_x','Unnamed: 0_y'],axis=1,inplace=True)
merge_11.to_csv('trainProcessed/loan_merge_11.csv')
merge_11_diff = merge_11[abs(merge_11['loan_sum'] - merge_11['11_loan_amount']) > 0.01]
merge_11_diff.to_csv('trainProcessed/loan_merge_11_diff.csv')