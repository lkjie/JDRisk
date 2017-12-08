import pandas as pd
import datetime
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

settings = EfficientFCParameters()

def run_click():
    t_click = pd.read_csv('../data/t_click.csv')
    try:
        t_click_8_10 = t_click[t_click['click_time'] < '2016-11-01']
        t_click_9_11 = t_click[t_click['click_time'] > '2016-08-31']
        extracted_features_click = extract_features(t_click_8_10, column_id="uid", column_sort="click_time",default_fc_parameters=settings)
        extracted_features_click.to_pickle('extracted_features_click_8_10.pickle')
        extracted_features_click = extract_features(t_click_9_11, column_id="uid", column_sort="click_time",default_fc_parameters=settings)
        extracted_features_click.to_pickle('extracted_features_click_9_11.pickle')
    except Exception as e:
        print(e)

def run_loan():
    t_loan = pd.read_csv('../data/t_loan.csv')
    try:
        t_loan_8_10 = t_loan[t_loan['loan_time']<'2016-11-01']
        t_loan_9_11 = t_loan[t_loan['loan_time']>'2016-08-31']
        extracted_features_loan = extract_features(t_loan_8_10, column_id="uid", column_sort="loan_time")
        extracted_features_loan.to_pickle('extracted_features_loan_8_10.pickle')
        extracted_features_loan = extract_features(t_loan_9_11, column_id="uid", column_sort="loan_time")
        extracted_features_loan.to_pickle('extracted_features_loan_9_11.pickle')
    except Exception as e:
        print(e)

def run_order():
    t_order = pd.read_csv('../data/t_order.csv')
    try:
        t_order = t_order.fillna(0)
        t_order_8_10 = t_order[t_order['buy_time']<'2016-11-01']
        t_order_9_11 = t_order[t_order['buy_time']>'2016-08-31']
        extracted_features_order = extract_features(t_order_8_10, column_id="uid", column_sort="buy_time",default_fc_parameters=settings)
        extracted_features_order.to_pickle('extracted_features_order_8_10.pickle')
        # extracted_features_order = extract_features(t_order_9_11, column_id="uid", column_sort="buy_time",default_fc_parameters=settings)
        # extracted_features_order.to_pickle('extracted_features_order_9_11.pickle')
    except Exception as e:
        print(e)

run_order()
# run_click()
# run_loan()