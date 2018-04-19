# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:26:47 2018

@author: Lone_kun
"""

import pandas as pd
import numpy as np
import time
train = pd.read_csv('../meinianData/train.csv', sep=',', encoding='gbk')
test = pd.read_csv('../meinianData/test.csv', sep=',', encoding='gbk')
merge_part1_2 = pd.read_csv('../meinianData/clean_1_data/merge_part12_delmiss.csv', sep=',', encoding='utf-8')
start_time = time.time()
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
train.to_csv('../meinianData/clean_1_data/train_set_1.csv', index = False, encoding = 'utf-8')
test.to_csv('../meinianData/clean_1_data/test_set_1.csv', index = False, encoding = 'utf-8')
print('total time:', time.time()-start_time)

# (38199, 394)    (9538, 394)



















