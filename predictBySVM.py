# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import os
import random

if (os.path.exists('../result') == False):
    os.mkdir('../result')

def loadData():
    train_x = pd.read_csv('../data/train_x_numeric_rank.csv')
    # train_x_category = pd.read_csv('../data/train_x_category_int.csv')
    # train_x = pd.merge(train_x_numeric,train_x_category,on='uid')
    train_x.drop('uid',axis=1)
    #对数据进行归一化
    train_x_norm = preprocessing.normalize(train_x)
    # train_x.to_csv('../data/train_x_svm.csv',index=None,encoding='utf-8')
    train_y = pd.read_csv('../data/train_y.csv')
    train_y = train_y.drop('uid',axis=1)

    test_x = pd.read_csv('../data/test_x_numeric_rank.csv')
    # test_x_category = pd.read_csv('../data/test_x_category_int.csv')
    # test_x = pd.merge(test_x_numeric,test_x_category,on='uid')
    test_uid = test_x.uid
    test_x.drop('uid',axis=1)
    #对数据进行归一化
    test_x_norm = preprocessing.normalize(test_x)
    y = []
    for i in range(len(train_y.values)):
        y.append(train_y.values[i][0])

    return train_x_norm,y,test_x_norm,test_uid

def predict_SVM():
    X,y,test_x,test_uid = loadData()
    model = SVC(probability=True)
    model.fit(X,y)
    test_y = model.predict_proba(test_x)
    result = pd.DataFrame(columns=["uid","score"])
    result.uid=test_uid
    result.score = test_y[:,1]
    result.to_csv('../result/result_SVM.csv',index=None,encoding='utf-8')

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
    C = [i/10 for i in range(10,50)]
    gamma = [i/1000.0 for i in range(1,51)]
    random.shuffle(random_seed)
    random.shuffle(C)
    random.shuffle(gamma)
    for i in range(50):
        pipeLine(i,C[i],gamma[i],random_seed[i])

def run():
    # return loadData()
    # predict_SVM()
    testSVM()




