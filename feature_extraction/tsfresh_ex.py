from tsfresh import extract_features
import pandas as pd
import datetime




# t_click = pd.read_csv('../data/t_click.csv')
t_loan = pd.read_csv('../data/t_loan.csv')
# t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
t_order = pd.read_csv('../data/t_order.csv')
# t_user = pd.read_csv('../data/t_user.csv')

t_order = t_order.fillna(0)
# extracted_features_click = extract_features(t_click, column_id="uid", column_sort="click_time")
# extracted_features_click.to_pickle('extracted_features_click.pickle')
extracted_features_loan = extract_features(t_loan, column_id="uid", column_sort="loan_time")
extracted_features_loan.to_pickle('extracted_features_loan.pickle')
extracted_features_order = extract_features(t_order, column_id="uid", column_sort="buy_time")
extracted_features_order.to_pickle('extracted_features_order.pickle')