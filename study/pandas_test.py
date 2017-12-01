import pandas as pd
import numpy as np
df1 = pd.DataFrame({'a':[1,2,1,2,1,2], 'b':[3,3,3,4,4,4], 'data':[12,13,11,8,10,3]})
# grouped = df1.groupby('b')
# 按照 'b' 这列分组了，name 为 'b' 的 key 值，group 为对应的df_group
# for name, group in grouped:
#     print(name, '->' , group)
# for x in grouped:
#     print(x)
grouped = df1.groupby(['a','b'])
# 按照 'b' 这列分组了，name 为 'b' 的 key 值，group 为对应的df_group
a = grouped.apply(lambda x:pd.DataFrame([1,2]))
for name, group in grouped:
    print (name, '->', group)