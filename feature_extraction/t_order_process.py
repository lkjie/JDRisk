import pandas as pd
import datetime
import numpy as np
# t_click = pd.read_csv('../data/t_click.csv')
# t_loan = pd.read_csv('../data/t_loan.csv')
# t_loan_sum = pd.read_csv('../data/t_loan_sum.csv')
# t_order = pd.read_csv('trainProcessed/t_order.csv')
t_order = pd.read_csv('../data/t_order.csv')
t_user = pd.read_csv('../data/t_user.csv')

feature = t_user[['uid']]
# 填充Nan值
t_order = t_order.fillna(0)
# t_order['exp_price'] = np.exp(t_order['price'])
# t_order.to_csv("trainProcessed/exp_t_order.csv")
# t_order = t_order.sort_values('price',ascending=False)
# t_order.to_csv("sorted_exp_order.csv")

'''
t_order
4个月购买商品总价，4个月购买商品总数，4个月购买商品每件均价，4个月享受优惠总次数，4个月享受优惠总价，4个月享受优惠均价
品类：（购物偏好）

'''
# 每个订单优惠金额

t_order['discount_price'] = t_order['price'] * t_order['discount']
# t_order['discount_exp_price'] = t_order['exp_price'] * t_order['discount']
t_order['month']=t_order['buy_time'].str.split('-').str.get(1)


# for month in t_order['month'].drop_duplicates().sort_values():
#     cols=['price','qty','discount','discount_exp_price']
#     feature[[str(month)+'_sum_'+str(c) for c in cols]] = t_order[t_order['month']==month].groupby('uid')[cols].sum()
#     feature[[str(month) + '_mean_' + str(c) for c in cols]] = t_order[t_order['month'] == month].groupby('uid')[cols].mean()

def feature_by_month():
    month = ['08', '09', '10', '11']
    for mt in month:
        feature_8 = feature
        t_order_8 = t_order[t_order['month'] == mt]
        cols = ['price', 'qty', 'discount']
        feature_8[['sum_' + str(c) for c in cols]] = t_order_8.groupby('uid')[cols].sum()
        feature_8[['mean_' + str(c) for c in cols]] = t_order_8.groupby('uid')[cols].mean()
        feature_8.to_csv('processedByMonth/order_%s_month.csv'%mt)


if __name__ == '__main__':
    feature_by_month()