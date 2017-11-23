import pandas as pd
import datetime
import numpy as np
t_click = pd.read_csv('../data/t_click.csv')
# t_loan = pd.read_csv('../data/t_loan.csv')
# t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
# t_order = pd.read_csv('../data/t_order.csv')
t_user = pd.read_csv('../data/t_user.csv')

'''
t_click
总点击次数，平均每天点击次数，平均每周点击次数，每月点击次数，
对某用户，如果点击次数在1分钟内，则合并为一次浏览（聚类），生成用户使用app的时间段数据，统计用户每天使用app的时长以及使用时间段

点击页面：（大家都关注的页面，与借贷数据合并，通过协同过滤或者关联规则寻找是否有浏览相同页面的人会产生类似的借贷趋势）
页面参数：
'''


t_click['month']=t_click['click_time'].str.split('-').str.get(1)
t_click['day']=t_click['click_time'].str.split(' ').str.get(0)
feature_click = t_user[['uid','age']]
for month in t_click['month'].drop_duplicates().sort_values():
    feature_click[[str(month)+'_month_sum_click']] = t_click[t_click['month']==month].groupby('uid')[['uid']].count()
for day in t_click['day'].drop_duplicates().sort_values():
    feature_click[[str(day)+'_day_sum_click']] = t_click[t_click['day']==day].groupby('uid')[['uid']].count()
    # 每天浏览app时长
    feature_click[str(day)+'day_duration'] = t_click[t_click['day']==day].groupby('uid').apply(lambda x:x,axis=1)
# 总点击次数
feature_click['all_click'] = t_click.groupby('uid')[['uid']].count()
# 平均每天点击次数
feature_click['day_click'] = feature_click['all_click'] / (datetime.datetime.strptime(t_click['day'].max(),"%Y-%m-%d") - datetime.datetime.strptime(t_click['day'].min(),"%Y-%m-%d")).days
# 平均每周点击次数
feature_click['week_click'] = feature_click['all_click'] / int(((datetime.datetime.strptime(t_click['day'].max(),"%Y-%m-%d") - datetime.datetime.strptime(t_click['day'].min(),"%Y-%m-%d")).days)/7)


print("")