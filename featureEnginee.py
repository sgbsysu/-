# -*- coding:utf-8 -*-
# 特征工程
# 对数值型特征进行rank排序，以增强对异常数据的鲁棒性，从而使模型更加稳定

import pandas as pd

train_x_csv = '../data/train_x.csv'
train_y_csv = '../data/train_y.csv'
test_x_csv = '../data/test_x.csv'
train_unlabeled_csv = '../data/train_unlabeled.csv'
features_type_csv = '../data/features_type.csv'

#从特征类型中，挑选出数值型的特征
features_type = pd.read_csv(features_type_csv)
numeric_feature = list(features_type[features_type.type == 'numeric'].feature)
category_feature = list(features_type[features_type.type == 'category'].feature)
def rank():
    train_x_numeric = pd.read_csv(train_x_csv)[['uid']+numeric_feature]
    train_x_numeric_rank = pd.DataFrame(train_x_numeric.uid,columns=['uid'])
    for feature in numeric_feature:
        train_x_numeric_rank['r'+feature] = train_x_numeric[feature].rank(method='max')
    train_x_numeric_rank.to_csv('../data/train_x_numeric_rank.csv',index=None,encoding='utf-8')

    test_x_numeric = pd.read_csv(test_x_csv)[['uid']+numeric_feature]
    test_x_numeric_rank = pd.DataFrame(test_x_numeric.uid,columns=['uid'])
    for feature in numeric_feature:
        test_x_numeric_rank['r'+feature] = test_x_numeric[feature].rank(method='max')
    test_x_numeric_rank.to_csv('../data/test_x_numeric_rank.csv',index=None,encoding='utf-8')

    train_unlabeled_numeric = pd.read_csv(train_unlabeled_csv)[['uid']+numeric_feature]
    train_unlabeled_numeric_rank = pd.DataFrame(train_unlabeled_numeric.uid,columns=['uid'])
    for feature in numeric_feature:
        train_unlabeled_numeric_rank['r'+feature] = train_unlabeled_numeric[feature].rank(method='max')
    train_unlabeled_numeric_rank.to_csv('../data/train_unlabeled_numeric_rank.csv',index=None,encoding='utf-8')

#对类型特征进行编码
def category2int():
    train_x_category = pd.read_csv(train_x_csv)[['uid']+category_feature]
    train_x_category_int = pd.DataFrame(train_x_category.uid,columns=['uid'])
    for feature in category_feature:
        train_x_category_int[feature] = (train_x_category[feature])
    train_x_category_int.to_csv('../data/train_x_category_int.csv',index=None,encoding='utf-8')

    test_x_category = pd.read_csv(test_x_csv)[['uid']+category_feature]
    test_x_category_int = pd.DataFrame(test_x_category.uid,columns=['uid'])
    for feature in category_feature:
        test_x_category_int[feature] = (test_x_category[feature])
    test_x_category_int.to_csv('../data/test_x_category_int.csv',index=None,encoding='utf-8')

    train_unlabeled_category = pd.read_csv(train_unlabeled_csv)[['uid']+category_feature]
    train_unlabeled_category_int = pd.DataFrame(train_unlabeled_category.uid,columns=['uid'])
    for feature in category_feature:
        train_unlabeled_category_int[feature] = (train_unlabeled_category[feature])
    train_unlabeled_category_int.to_csv('../data/train_unlabeled_category_int.csv',index=None,encoding='utf-8')

#离散特征，在排序特征的基础上，对feature的值进行划分
#我们采用10个区间的方式进行划分
def discrete():
    train_x = pd.read_csv('../data/train_x_numeric_rank.csv')
    featureList = list(train_x.columns.values)
    featureList.remove('uid')
    train_x_discrete = pd.DataFrame(train_x.uid,columns=['uid'])
    #离散化
    train_x[train_x<1500] = 1
    train_x[(train_x>=1500)&(train_x<3000)] = 2
    train_x[(train_x>=3000)&(train_x<4500)] = 3
    train_x[(train_x>=4500)&(train_x<6000)] = 4
    train_x[(train_x>=6000)&(train_x<7500)] = 5
    train_x[(train_x>=7500)&(train_x<9000)] = 6
    train_x[(train_x>=9000)&(train_x<10500)] = 7
    train_x[(train_x>=10500)&(train_x<12000)] = 8
    train_x[(train_x>=12000)&(train_x<13500)] = 9
    train_x[train_x>=13500] = 10
    for feature in featureList:
        train_x_discrete['d'+feature] = train_x[feature]
    train_x_discrete.to_csv('../data/train_x_discrete.csv',index=None,encoding='utf-8')

    test_x = pd.read_csv('../data/test_x_numeric_rank.csv')
    featureList_t = list(test_x.columns.values)
    featureList_t.remove('uid')
    test_x_discrete = pd.DataFrame(test_x.uid,columns=['uid'])
    #离散化
    test_x[test_x<500] = 1
    test_x[(test_x>=500)&(test_x<1000)] = 2
    test_x[(test_x>=1000)&(test_x<1500)] = 3
    test_x[(test_x>=1500)&(test_x<2000)] = 4
    test_x[(test_x>=2000)&(test_x<2500)] = 5
    test_x[(test_x>=2500)&(test_x<3000)] = 6
    test_x[(test_x>=3000)&(test_x<3500)] = 7
    test_x[(test_x>=3500)&(test_x<4000)] = 8
    test_x[(test_x>=4000)&(test_x<4500)] = 9
    test_x[test_x>=4500] = 10
    for feature in featureList_t:
        test_x_discrete['d'+feature] = test_x[feature]
    test_x_discrete.to_csv('../data/test_x_discrete.csv',index=None,encoding='utf-8')

    train_unlabeled = pd.read_csv('../data/train_unlabeled_numeric_rank.csv')
    featureList_u = list(train_unlabeled.columns.values)
    featureList_u.remove('uid')
    train_unlabeled_discrete = pd.DataFrame(train_unlabeled.uid,columns=['uid'])
    #离散化
    train_unlabeled[train_unlabeled<5000] = 1
    train_unlabeled[(train_unlabeled>=5000)&(train_unlabeled<10000)] = 2
    train_unlabeled[(train_unlabeled>=10000)&(train_unlabeled<15000)] = 3
    train_unlabeled[(train_unlabeled>=15000)&(train_unlabeled<20000)] = 4
    train_unlabeled[(train_unlabeled>=20000)&(train_unlabeled<25000)] = 5
    train_unlabeled[(train_unlabeled>=25000)&(train_unlabeled<30000)] = 6
    train_unlabeled[(train_unlabeled>=30000)&(train_unlabeled<35000)] = 7
    train_unlabeled[(train_unlabeled>=35000)&(train_unlabeled<40000)] = 8
    train_unlabeled[(train_unlabeled>=40000)&(train_unlabeled<45000)] = 9
    train_unlabeled[train_unlabeled>=45000] = 10
    for feature in featureList_u:
        train_unlabeled_discrete['d'+feature] = train_unlabeled[feature]
    train_unlabeled_discrete.to_csv('../data/train_unlabeled_discrete.csv',index=None,encoding='utf-8')

def run():
    rank()
    discrete()
    # category2int()




