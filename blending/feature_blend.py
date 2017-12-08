import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

'''
add tsfresh
'''

ts_loan = pd.read_pickle('/home/share/JD/Loan_Forecasting_Qualification/feature_extraction/extracted_features_loan.pickle')

test = pd.read_csv('../model_file/test_feature_1201.csv')
train = pd.read_csv('../model_file/train_feature_1201.csv')

test = pd.merge(test, ts_loan,how='left',left_on='uid',right_index=True)
train = pd.merge(train, ts_loan,how='left',left_on='uid',right_index=True)

test.fillna(0,inplace=True)
train.fillna(0,inplace=True)
test.to_csv('../model_file/test_feature_1208.csv',index=False)
train.to_csv('../model_file/train_feature_1208.csv',index=False)

