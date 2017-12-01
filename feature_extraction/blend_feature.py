import pandas as pd
import datetime
import numpy as np

test_feature = pd.read_csv('processedByMonth/user.csv').drop('Unnamed: 0',axis=1)
train_feature = pd.read_csv('processedByMonth/user.csv').drop('Unnamed: 0',axis=1)
train_month = ['08','09','10']
test_month = ['09','10','11']
for mt in train_month:
    click = pd.read_csv('processedByMonth/click_%s_month.csv'%mt).drop('Unnamed: 0',axis=1)
    loan = pd.read_csv('processedByMonth/loan_%s_month.csv'%mt).drop('Unnamed: 0',axis=1)
    order = pd.read_csv('processedByMonth/order_%s_month.csv'%mt).drop('Unnamed: 0',axis=1)
    train_feature = pd.merge(train_feature, click,'left',on='uid')
    train_feature = pd.merge(train_feature, loan, 'left', on='uid')
    train_feature = pd.merge(train_feature, order, 'left', on='uid')
# 数据处理和清洗
train_feature['active_date'] = pd.to_datetime(train_feature['active_date'])
train_feature['active_date'] = train_feature.active_date.values.astype(np.int64)
train_feature = train_feature.fillna(0)
print(train_feature.columns)
print(train_feature.shape)
train_feature.to_csv('../model_file/train_feature_1201.csv',index=False)

for mt in test_month:
    click = pd.read_csv('processedByMonth/click_%s_month.csv'%mt).drop('Unnamed: 0',axis=1)
    loan = pd.read_csv('processedByMonth/loan_%s_month.csv'%mt).drop('Unnamed: 0',axis=1)
    order = pd.read_csv('processedByMonth/order_%s_month.csv'%mt).drop('Unnamed: 0',axis=1)
    test_feature = pd.merge(test_feature, click,'left',on='uid')
    test_feature = pd.merge(test_feature, loan, 'left', on='uid')
    test_feature = pd.merge(test_feature, order, 'left', on='uid')
test_feature['active_date'] = pd.to_datetime(test_feature['active_date'])
test_feature['active_date'] = test_feature.active_date.values.astype(np.int64)
test_feature = test_feature.fillna(0)
print(test_feature.columns)
print(test_feature.shape)
test_feature.to_csv('../model_file/test_feature_1201.csv',index=False)