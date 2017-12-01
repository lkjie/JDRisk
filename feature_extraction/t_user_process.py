import pandas as pd
import datetime
import numpy as np
# t_click = pd.read_csv('../data/t_click.csv')
# t_loan = pd.read_csv('../data/t_loan.csv')
# t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
# t_order = pd.read_csv('../data/t_order.csv')
t_user = pd.read_csv('../data/t_user.csv')

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

feature = t_user
feature['age_range'] = 0
feature.loc[feature['age'] <= 22, 'age_range'] = 1
feature.loc[(22 < feature['age']) & (feature['age'] <= 28), 'age_range'] = 2
feature.loc[(28 < feature['age']) & (feature['age'] <= 35), 'age_range'] = 3
feature.loc[(35 < feature['age']) & (feature['age'] <= 45), 'age_range'] = 4
feature.loc[feature['age'] > 45, 'age_range'] = 5
feature.to_csv('processedByMonth/user.csv')