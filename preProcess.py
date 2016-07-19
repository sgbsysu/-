# -*- coding:utf-8 -*-
# 数据预处理
# 对原始数据的缺失值进行分析
# 统计每个变量缺失值的数目并可视化

import pandas as pd
import matplotlib.pyplot as plt

train_x_csv = '../data/train_x.csv'
train_y_csv = '../data/train_y.csv'
test_x_csv = '../data/test_x.csv'
train_unlabeled_csv = '../data/train_unlabeled.csv'

#统计每个样本，缺失值的数目
def countNull():
    #将每行样本的缺失值存储在train_x_null['n_null']中
    train_x = pd.read_csv(train_x_csv)
    train_x['n_null'] = (train_x < 0).sum(axis=1)
    train_x.to_csv('../data/train_x_null.csv',index=None,encoding='utf-8')
    #对test_x,train_unlabeled进行同样的统计
    test_x = pd.read_csv(test_x_csv)
    test_x['n_null'] = (test_x < 0).sum(axis=1)
    test_x.to_csv('../data/test_x_null.csv',index=None,encoding='utf-8')
    train_unlabeled = pd.read_csv(train_unlabeled_csv)
    train_unlabeled['n_null'] = (train_unlabeled < 0).sum(axis=1)
    train_unlabeled.to_csv('../data/train_unlabeled_null.csv',index=None,encoding='utf-8')

#对样本缺失值进行可视化处理
def visualNull():
    train_x = pd.read_csv('../data/train_x_null.csv')[['uid','n_null']]
    test_x = pd.read_csv('../data/test_x_null.csv')[['uid','n_null']]
    train_unlabeled = pd.read_csv('../data/train_unlabeled_null.csv')[['uid','n_null']]

    train_x = train_x.sort(columns='n_null')
    test_x = test_x.sort(columns='n_null')
    train_unlabeled = train_unlabeled.sort(columns='n_null')
    y_train_x = train_x.n_null.values
    x_train_x = range(len(y_train_x))
    plt.scatter(x_train_x,y_train_x,c='k')
    plt.title('train set')
    plt.show()

    y_test_x = test_x.n_null.values
    x_test_x = range(len(y_test_x))
    plt.scatter(x_test_x,y_test_x,c='k')
    plt.title('test set')
    plt.show()

    y_train_unlabeled = train_unlabeled.n_null.values
    x_train_unlabeled = range(len(y_train_unlabeled))
    plt.scatter(x_train_unlabeled,y_train_unlabeled,c='k')
    plt.title('train unlabeled')
    plt.show()

#根据缺失值所得到的散点图，对其进行分段处理
#查看散点图，可估计：
#0~35；35~69；69~145；145~190；190~
def discreteNull():
    train_x = pd.read_csv('../data/train_x_null.csv')
    train_x['discrete_null'] = train_x.n_null
    train_x.discrete_null[train_x.discrete_null<=35] = 1
    train_x.discrete_null[(train_x.discrete_null>35)&(train_x.discrete_null<=69)] = 2
    train_x.discrete_null[(train_x.discrete_null>69)&(train_x.discrete_null<=145)] = 3
    train_x.discrete_null[(train_x.discrete_null>145)&(train_x.discrete_null<=190)] = 4
    train_x.discrete_null[train_x.discrete_null>190] = 5
    train_x.to_csv('../data/train_x_discrete_null.csv',index=None,encoding='utf=8')

    test_x = pd.read_csv('../data/test_x_null.csv')
    test_x['discrete_null'] = test_x.n_null
    test_x.discrete_null[test_x.discrete_null<=35] = 1
    test_x.discrete_null[(test_x.discrete_null>35)&(test_x.discrete_null<=69)] = 2
    test_x.discrete_null[(test_x.discrete_null>69)&(test_x.discrete_null<=145)] = 3
    test_x.discrete_null[(test_x.discrete_null>145)&(test_x.discrete_null<=190)] = 4
    test_x.discrete_null[test_x.discrete_null>190] = 5
    test_x.to_csv('../data/test_x_discrete_null.csv',index=None,encoding='utf-8')

    train_unlabeled = pd.read_csv('../data/train_unlabeled_null.csv')
    train_unlabeled['discrete_null'] = train_unlabeled.n_null
    train_unlabeled.discrete_null[train_unlabeled.discrete_null<=35] = 1
    train_unlabeled.discrete_null[(train_unlabeled.discrete_null>35)&(train_unlabeled.discrete_null<=69)] = 2
    train_unlabeled.discrete_null[(train_unlabeled.discrete_null>69)&(train_unlabeled.discrete_null<=145)] = 3
    train_unlabeled.discrete_null[(train_unlabeled.discrete_null>145)&(train_unlabeled.discrete_null<=190)] = 4
    train_unlabeled.discrete_null[train_unlabeled.discrete_null>190] = 5
    train_unlabeled.to_csv('../data/train_unlabeled_discrete_null.csv',index=None,encoding='utf-8')

def run():
    # countNull()
    # visualNull()
    discreteNull()


