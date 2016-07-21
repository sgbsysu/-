# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import os
import random

if (os.path.exists('../result_svm') == False):
    os.mkdir('../result_svm')

#加载文件路径
train_xy_csv = '../data_model/train_xy.csv'
test_x_csv = '../data_model/test_x.csv'

def loadData():
    train_xy = pd.read_csv(train_xy_csv)
    train_x = train_xy.drop(['uid','y'],axis=1)
    #对数据进行归一化
    train_x_norm = preprocessing.normalize(train_x)
    train_y = train_xy.y

    test = pd.read_csv(test_x_csv)
    test_uid = test.uid
    test_x = test.drop('uid',axis=1)
    #对数据进行归一化
    test_x_norm = preprocessing.normalize(test_x)
    y = list(train_y.values)

    return train_x_norm,y,test_x_norm,test_uid

# def predict_SVM():
#     X,y,test_x,test_uid = loadData()
#     model = SVC(probability=True)
#     model.fit(X,y)
#     test_y = model.predict_proba(test_x)
#     result = pd.DataFrame(columns=["uid","score"])
#     result.uid=test_uid
#     result.score = test_y[:,1]
#     result.to_csv('../result/result_SVM_2.csv',index=None,encoding='utf-8')

def pipeLine(iteration,C,gamma,random_seed):
    X,y,test_x,test_uid = loadData()
    model = SVC(C=C,kernel='rbf',gamma=gamma,probability=True,random_state=random_seed)
    model.fit(X,y)
    pred = model.predict_proba(test_x)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid=test_uid
    test_result.score = pred[:,1]
    test_result.to_csv('../result/svm_pred{0}.csv'.format(iteration),index=None,encoding='utf-8')

def testSVM():
    random_seed = range(1024,2048)
    C = [i/10 for i in range(10,20)]
    gamma = [i/1000.0 for i in range(1,11)]
    random.shuffle(random_seed)
    random.shuffle(C)
    random.shuffle(gamma)
    for i in range(10):
        pipeLine(i,C[i],gamma[i],random_seed[i])

testSVM()





