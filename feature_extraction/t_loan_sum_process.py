import pandas as pd
import datetime
import numpy as np
# t_click = pd.read_csv('../data/t_click.csv')
# t_loan = pd.read_csv('../data/t_loan.csv')
t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
# t_order = pd.read_csv('../data/t_order.csv')
t_user = pd.read_csv('../data/t_user.csv')

t_loan_sum['exp_loan_sum'] = np.exp(t_loan_sum['loan_sum'])
t_loan_sum.to_csv('trainProcessed/exp_t_loan_sum.csv')