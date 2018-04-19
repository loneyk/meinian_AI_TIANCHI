# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:10:42 2018

@author: Lone_kun
"""

import pandas as pd
import time
#读取数据
#实现的功能：对同一病人体检超过一次的项目，进行数据拼接
#train = pd.read_csv('../meinianData/train.csv',encoding='UTF-8')
#test = pd.read_csv('../meinianData/test.csv',encoding='ISO-8859-1')
part_1 = pd.read_csv('../meinianData/meinian_round1_data_part1_20180408.txt', sep='$')
part_2 = pd.read_csv('../meinianData/meinian_round1_data_part2_20180408.txt', sep='$')
part_1_2 = pd.concat([part_1, part_2])
part_1_2 = pd.DataFrame(part_1_2).sort_values('﻿vid').reset_index(drop=True)
begin_time = time.time()
print('begin')
# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0]>1 :
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df
# 数据简单处理
print('find_is_copy')
print(part_1_2.shape)
is_happen = part_1_2.groupby(['﻿vid', 'table_id']).size().reset_index()#以前两项为索引，统计总数（不重复）,size那一列的列名为0
# 重塑index用来去重
is_happen['new_index'] = is_happen['﻿vid'] + '_' + is_happen['table_id']
is_happen_new = is_happen[is_happen[0]>1]['new_index'] #统计病人体检超过一次的项目

part_1_2['new_index'] = part_1_2['﻿vid'] + '_' + part_1_2['table_id']

repeat_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
repeat_part = repeat_part.sort_values(['﻿vid', 'table_id'])#repeat_part也有四列
non_repeat_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]

print('begin')
part_1_2_repeat_part = repeat_part.groupby(['﻿vid', 'table_id']).apply(merge_table).reset_index()
part_1_2_repeat_part.rename(columns={0:'field_results'}, inplace = True)
print('......')
tmp = pd.concat([part_1_2_repeat_part, non_repeat_part[['﻿vid','table_id','field_results']]])
#行列转换
print('finish')
tmp = tmp.pivot(index = '﻿vid', values = 'field_results', columns = 'table_id')#设置行列值
#tmp.to_csv('../meinianData/tmp1.csv)
tmp.to_csv('../meinianData/tmp1.csv')
print(tmp.shape)
print('total time:',time.time()-begin_time)
























