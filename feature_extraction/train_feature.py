import pandas as pd
import datetime
import numpy as np
t_click = pd.read_csv('../data/t_click.csv')
t_loan = pd.read_csv('../data/t_loan.csv')
t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
t_order = pd.read_csv('../data/t_order.csv')
t_user = pd.read_csv('../data/t_user.csv')

'''
提取8,9,10三个月的特征，预测11月的借贷总额
'''

'''
测试集：uid，时间，特征
回归问题，利用8月份特征训练九月份借款总额，以此类推
'''
feature = t_user

# loan
# 每月借款总额，每周借款总额，每月借款次数，每周借款次数，4个月借款总额，8-11月每月平均借款，8-10月每月平均借款（排除双十一），借款总期数，每次借款平均期数，每月借款平均期数
'''
2016-08-03(周三)到2016-11-30(周三)
'''
t_loan_8 = t_loan[(t_loan['loan_time']>="2016-08-01 00:00:00") & (t_loan['loan_time']<"2016-09-01 00:00:00")]
t_loan_8_sum = t_loan_8.groupby('uid').sum()
t_loan_8_sum.rename(columns=lambda x:'1_'+x, inplace=True)  # 8月份，第一个月
t_loan_8_sum.reset_index(level=['uid'], inplace=True)
t_loan_9 = t_loan[(t_loan['loan_time']>="2016-09-01 00:00:00") & (t_loan['loan_time']<"2016-10-01 00:00:00")]
t_loan_9_sum = t_loan_9.groupby('uid').sum()
t_loan_9_sum.rename(columns=lambda x:'2_'+x, inplace=True) # 9月份，第2个月
t_loan_9_sum.reset_index(level=['uid'], inplace=True)
t_loan_10 = t_loan[(t_loan['loan_time']>="2016-10-01 00:00:00") & (t_loan['loan_time']<"2016-11-01 00:00:00")]
t_loan_10_sum = t_loan_10.groupby('uid').sum()
t_loan_10_sum.rename(columns=lambda x:'3_'+x, inplace=True) # 10月份，第3个月
t_loan_10_sum.reset_index(level=['uid'], inplace=True)

feature = pd.merge(feature, t_loan_8_sum, 'left', left_on='uid', right_on='uid')
feature = pd.merge(feature, t_loan_9_sum, 'left', left_on='uid', right_on='uid')
feature = pd.merge(feature, t_loan_10_sum, 'left', left_on='uid', right_on='uid')
# 按周聚合数据
d_start = datetime.datetime.strptime('2016-08-03 00:00:00', '%Y-%m-%d %H:%M:%S')
d_end = datetime.datetime.strptime('2016-11-02 00:00:00', '%Y-%m-%d %H:%M:%S')
d_now = d_start
week = datetime.timedelta(days=7)
t_loan['loan_time'] = pd.to_datetime(t_loan['loan_time'])
week_count = 1
while d_now < d_end:
    print(d_now.strftime("%Y-%m-%d")+'    '+(d_now+week).strftime("%Y-%m-%d"))
    # 每周借款总数
    t_loan_this_week = t_loan[(t_loan['loan_time'] >= d_now) & (t_loan['loan_time'] < d_now + week)].groupby(
        'uid').sum()
    # 每周借款次数
    t_loan_this_week_count = t_loan[(t_loan['loan_time'] >= d_now) & (t_loan['loan_time'] < d_now + week)].groupby(
        'uid').count()
    # t_loan_this_week.rename(columns=lambda x: 'week_' + d_now.strftime('%m-%d') + x, inplace=True)
    t_loan_this_week.rename(columns=lambda x: 'week_' + str(week_count) + x, inplace=True)
    feature = pd.merge(feature, t_loan_this_week, 'left', left_on='uid', right_index=True)
    d_now += week
    week_count += 1

'''
t_user
年龄：学生（<=22岁），青年（23-28岁），结婚（29-35岁），中年（36-45岁），中年以上（>=46岁）
初始额度：
激活日期：按年分组

'''


def f_split_age(x):
    if x['age'] <= 22:
        x['age_range'] = 1
    elif 22 < x['age'] <= 28:
        x['age_range'] = 2
    elif 28 < x['age'] <= 35:
        x['age_range'] = 3
    elif 35 < x['age'] <= 45:
        x['age_range'] = 4
    else:
        x['age_range'] = 5


feature['age_range'] = 0
feature.loc[feature['age'] <= 22, 'age_range'] = 1
feature.loc[(22 < feature['age']) & (feature['age'] <= 28), 'age_range'] = 2
feature.loc[(28 < feature['age']) & (feature['age'] <= 35), 'age_range'] = 3
feature.loc[(35 < feature['age']) & (feature['age'] <= 45), 'age_range'] = 4
feature.loc[feature['age'] > 45, 'age_range'] = 5

'''
t_order
4个月购买商品总价，4个月购买商品总数，4个月购买商品每件均价，4个月享受优惠总次数，4个月享受优惠总价，4个月享受优惠均价
品类：（购物偏好）

'''

t_order['month']=t_order['buy_time'].str.split('-').str.get(1)

for i,month in enumerate(t_order['month'].drop_duplicates().sort_values()):
    if month == '11':
        continue
    cols=['price','qty','discount'	]
    feature[[str(i)+'_'+str(c) for c in cols]] = t_order[t_order['month']==month].groupby('uid')[cols].sum()
'''
t_click
总点击次数，平均每天点击次数，平均每周点击次数，平均每月点击次数，
对某用户，如果点击次数在1分钟内，则合并为一次浏览（聚类），生成用户使用app的时间段数据，统计用户每天使用app的时长以及使用时间段

点击页面：（大家都关注的页面，与借贷数据合并，通过协同过滤或者关联规则寻找是否有浏览相同页面的人会产生类似的借贷趋势）
页面参数：
'''

'''
t_loan_sum

'''


feature['active_date'] = pd.to_datetime(feature['active_date'])
feature['active_date'] = feature.active_date.values.astype(np.int64)
feature.to_csv('../model_file/feature08_10.csv')
print("")