import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import numpy as np

# 训练特征文件
f_train = '../model_file/train_feature_1201.csv'
# 预测特征文件
f_predict = '../model_file/test_feature_1201.csv'
# 提交结果文件
f_submit = 'uploadfile/1201.csv'

# 测试集比例
test_size=0.2

num_round = 16
param = {
    'max_depth': 2,
    'eta': 1,
    'silent': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse'
}

feature = pd.read_csv(f_train)
feature.columns = ['uid']+ [i for i in range(feature.shape[1] -1)]
label = pd.read_csv('../data/t_loan_sum.csv', usecols=['uid', 'loan_sum'])
data = feature.merge(label, on='uid')
# data.fillna(0,inplace=True)
data = data.set_index('uid')

train, test = train_test_split(data, test_size=test_size, random_state=1)

feature_cols = train.columns.drop('loan_sum')
dtrain = xgb.DMatrix(train[feature_cols], train['loan_sum'])
dtest = xgb.DMatrix(test[feature_cols], test['loan_sum'])

bst = xgb.train(param, dtrain, num_round)
print(bst.eval(dtrain))
print(bst.eval(dtest))
num_round = 16

print(bst.get_score())

dtrain = xgb.DMatrix(data[data.columns.drop('loan_sum')], data['loan_sum'])
bst = xgb.train(param, dtrain, num_round)
print(bst.eval(dtrain))
print(bst.get_score())

predict = pd.read_csv(f_predict, index_col='uid')
predict.columns = [i for i in range(predict.shape[1])]
dpredict = xgb.DMatrix(predict[feature_cols])

submit = predict.index.to_frame()
submit['predict']=bst.predict(dpredict)
submit.to_csv(f_submit,header=False,index=False)
print('write submit file to '+f_submit)