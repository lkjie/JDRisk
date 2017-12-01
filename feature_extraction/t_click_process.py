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

def all_handel():
    t_click = pd.read_csv('../data/t_click.csv')
    t_click['month'] = t_click['click_time'].str.split('-').str.get(1)
    t_click['day'] = t_click['click_time'].str.split(' ').str.get(0)
    t_click = t_click.sort_values('click_time')
    t_click['click_time'] = pd.to_datetime(t_click['click_time'])

    # 每天浏览app时长
    t_click['run_number'] = t_click.groupby('uid')['click_time'].apply(
        lambda s: (s - s.shift(1) > datetime.timedelta(minutes=3)).fillna(0).cumsum(skipna=False))
    df_duration = t_click.groupby(['uid', 'run_number'], group_keys=False)['click_time'].apply(
        lambda x: x.max() - x.min())
    df_duration = pd.DataFrame(df_duration).reset_index()
    df_duration.rename(columns={'click_time': 'duration'}, inplace=True)
    df_duration.to_csv('midProcessed/t_click_duration.csv')
    # t_click = pd.merge(t_click,df_duration,'left',['uid','run_number'])

    feature_click = t_user[['uid', 'age']]
    for month in t_click['month'].drop_duplicates().sort_values():
        feature_click[[str(month) + '_month_sum_click']] = t_click[t_click['month'] == month].groupby('uid')[
            ['uid']].count()
    for day in t_click['day'].drop_duplicates().sort_values():
        feature_click[[str(day) + '_day_sum_click']] = t_click[t_click['day'] == day].groupby('uid')[['uid']].count()
    # 总点击次数
    feature_click['all_click'] = t_click.groupby('uid')[['uid']].count()
    # 平均每天点击次数
    feature_click['day_click'] = feature_click['all_click'] / (
    datetime.datetime.strptime(t_click['day'].max(), "%Y-%m-%d") - datetime.datetime.strptime(t_click['day'].min(),
                                                                                              "%Y-%m-%d")).days
    # 平均每周点击次数
    feature_click['week_click'] = feature_click['all_click'] / int(((datetime.datetime.strptime(t_click['day'].max(),
                                                                                                "%Y-%m-%d") - datetime.datetime.strptime(
        t_click['day'].min(), "%Y-%m-%d")).days) / 7)

    # 测试双十一影响
    # 按人和日期排序测分布
    # for day in t_click['day'].drop_duplicates().sort_values():
    #     feature_click[[str(day)+'_day_sum_click']] = t_click[t_click['day']==day].groupby('uid')[['uid']].count()
    print(t_click)
    t_click.to_csv('midProcessed/t_click_feature.csv')


def feature_by_month():
    month = [8,9,10,11]
    # 分割8-10月与9-11月的数据
    df_duration = pd.read_csv('midProcessed/t_click_duration.csv')
    t_click = pd.read_csv('midProcessed/t_click_feature.csv')
    t_click = pd.merge(t_click,df_duration,'left',['uid','run_number'])
    feature = pd.read_csv('../data/t_user.csv')[['uid']]
    t_click['duration'] = pd.to_timedelta(t_click['duration'])
    for mt in month:
        # 按月特征
        feature_8 = feature
        t_click_8 = t_click[t_click['month'] == mt]
        feature_8['all_click'] = t_click_8.groupby('uid')[['uid']].count()
        feature_8['day_click'] = feature_8['all_click'] / (datetime.datetime.strptime(t_click['day'].max(),"%Y-%m-%d") - datetime.datetime.strptime(t_click['day'].min(),"%Y-%m-%d")).days
        feature_8['week_click'] = feature_8['all_click'] / int(((datetime.datetime.strptime(t_click['day'].max(),"%Y-%m-%d") - datetime.datetime.strptime(t_click['day'].min(),"%Y-%m-%d")).days)/7)
        # 按月浏览app总时长
        feature_8['duration'] = t_click_8.groupby(['uid'])['duration'].sum()
        # 按分钟化为整形
        feature_8['duration'] = feature_8['duration'].apply(lambda x:x.seconds / 60)
        feature_8.to_csv('processedByMonth/click_%02d_month.csv'%mt)

if __name__=='__main__':
    feature_by_month()