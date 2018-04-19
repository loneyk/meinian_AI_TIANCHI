# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:20:02 2018

@author: Lone_kun
"""

import pandas as pd
import numpy as np
import time
#from base_func import remain_feat
start_time = time.time()
#读取数据
train = pd.read_csv('../meinianData/train.csv', sep=',', encoding='gbk')
test = pd.read_csv('../meinianData/test.csv', sep=',', encoding='gbk')
data_part1 = pd.read_csv('../meinianData/meinian_round1_data_part1_20180408.txt',sep='$',encoding='utf-8')
data_part2 = pd.read_csv('../meinianData/meinian_round1_data_part2_20180408.txt', sep='$', encoding='utf-8')

# data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
part1_2 = pd.concat([data_part1, data_part2], axis=0) #{0/'index', 1/'columns'}, default 0
part1_2 = pd.DataFrame(part1_2).sort_values('﻿vid').reset_index(drop=True)
vid_set = pd.concat([train['vid'], test['vid']],axis=0)
vid_set = pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
part1_2 = part1_2[part1_2['﻿vid'].isin(vid_set['vid'])]
                 
# 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
def filter_None(data):
    data=data[data['field_results']!='']
    data=data[data['field_results']!='未查']
    return data
part1_2 = filter_None(part1_2)
# 过滤列表，过滤掉不重要的table_id 所在行
filter_list=['0203','0209','0702','0703','0705','0706','0709','0726','0730','0731','3601',
             '1308','1316']
part1_2 = part1_2[~part1_2['table_id'].isin(filter_list)]
# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0]>1 :
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df
#数据简单处理
print(part1_2.shape)
vid_tabid_group = part1_2.groupby(['﻿vid', 'table_id']).size().reset_index()
# 重塑index用来去重,区分重复部分和唯一部分
print('去重和组合')
vid_tabid_group['new_index'] = vid_tabid_group['﻿vid'] + '_' + vid_tabid_group['table_id']
vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0]>1]['new_index']
# print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
part1_2['new_index'] = part1_2['﻿vid'] + '_' + part1_2['table_id']

dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
dup_part = dup_part.sort_values(['﻿vid', 'table_id'])  
unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))] 
                      
part1_2_dup = dup_part.groupby(['﻿vid', 'table_id']).apply(merge_table).reset_index()          
part1_2_dup.rename(columns={0:'field_results'},inplace=True)   
part1_2_res = pd.concat([part1_2_dup, unique_part[['﻿vid', 'table_id', 'field_results']]])  

table_id_group = part1_2.groupby('table_id').size().sort_values(ascending=False)    
table_id_group.to_csv('../meinianData/clean_1_data/part_tabid_size.csv', encoding='utf-8')   

#行列转换
print('重新组织index和columns')
merge_part1_2 = part1_2_res.pivot(index='﻿vid', values='field_results', columns='table_id') 
print('new part finished')
print(merge_part1_2.shape)
merge_part1_2.to_csv('../meinianData/clean_1_data/merge_part1_2.csv', encoding='utf-8')
print(merge_part1_2.head())
del merge_part1_2
                   
time.sleep(10)
print('重新读取数据...')
merge_part1_2 = pd.read_csv('../meinianData/clean_1_data/merge_part1_2.csv',sep=',',encoding='utf-8')

# 删除掉一些出现次数低，缺失比例大的字段，保留超过阈值的特征
def remain_feat(df, thresh=0.9):
    exclude_feats = []
    print('移除数据缺失多的字段')
    print('移除之前的数量：', len(df.columns))
    num_rows = df.shape[0]
    for c in df.columns:
        num_missing = df[c].isnull().sum()
        if num_missing==0:
            continue
        missing_percent = num_missing / float(num_rows)
        if missing_percent>thresh:
            exclude_feats.append(c)
        print("移除缺失数据的字段量： %s" % len(exclude_feats))
    # 保留超过阈值的特征
    feats = []
    for c in df.columns:
        if c not in exclude_feats:
            feats.append(c)
    print('剩余的字段数量：', len(feats))
    return feats
    
feats = remain_feat(merge_part1_2, thresh=0.96)

merge_part1_2 = merge_part1_2[feats]
merge_part1_2.to_csv('../meinianData/clean_1_data/merge_part12_delmiss.csv',encoding='utf-8')


#clean_train_label拼接
#统一格式
train.rename(columns={'vid':'vid'},inplace=True)
test.rename(columns={'vid':'vid'},inplace=True)
merge_part1_2.rename(columns={'﻿vid':'vid'},inplace=True)

train_of_part = merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
test_of_part = merge_part1_2[merge_part1_2['vid'].isin(test['vid'])] 
#预测结果和标签拼接
train = pd.merge(train, train_of_part, on = 'vid')    
test = pd.merge(test, test_of_part, on = 'vid')

#清洗训练集中的指标
def clean_label(x):
    x = str(x)
    if '+' in x:   #16.04++
        i = x.index('+')
        x=x[0:i]    
    if '>' in x:    #> 11.00
        i = x.index('>')
        x=x[i+1:]
    if len(x.split(sep='.'))>2:#2.2.8
        i = x.rindex('.')
        x = x[0:i]+x[i+1:]
    if '未做' in x or '未查' in x or '弃查' in x:
        x = np.nan
    if str(x).isdigit() == False and len(str(x))>4:
        x=x[0:4]
    return x
    

def data_clean(df):
    for c in ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']:
        df[c] = df[c].apply(clean_label)
        df[c] = df[c].astype('float64')
    return df
    
train = data_clean(train)

print('Save train_set and test_set')
train.to_csv('../meinianData/clean_1_data/train_set_2.csv', index = False, encoding = 'utf-8')
test.to_csv('../meinianData/clean_1_data/test_set_2.csv', index = False, encoding = 'utf-8')

print('total time:', time.time()-start_time)

# train.shape   (38199, 393)
#test.shape    (9538, 393)                        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    