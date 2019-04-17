import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time
from xgboost import XGBClassifier

def decicionTest(tra_X, tra_Y, tes_X, tes_Y):
    dtree = DecisionTreeClassifier()
    dtree.fit(tra_X,tra_Y)
    predictions = dtree.predict(tes_X)
    report = classification_report(tes_Y,predictions)
    matrix = confusion_matrix(tes_Y,predictions)
    score = f1_score_1(report)
    return score, [predictions, report, matrix]
def r_d_forestTest(tra_X, tra_Y, tes_X, tes_Y):
    rdf = RandomForestClassifier(n_estimators=100)
    rdf.fit(tra_X,tra_Y)
    predictions = rdf.predict(tes_X)
    report = classification_report(tes_Y,predictions)
    matrix = confusion_matrix(tes_Y,predictions)
    score = f1_score_1(report)
    return score, [predictions, report, matrix]
def xgboostTest(tra_X, tra_Y, tes_X, tes_Y):
    xgb = XGBClassifier()
    xgb.fit(tra_X,tra_Y)
    y_pred = xgb.predict(tes_X)
    predictions = [round(value) for value in y_pred]
    report = classification_report(tes_Y,predictions)
    matrix = confusion_matrix(tes_Y,predictions)
    score = f1_score_1(report)
    return score, [predictions, report, matrix]

def selector(df, col, min_date, max_date):
    return df[(df[col]>=min_date) & (df[col]<=max_date)]
def cmm(d, key, variable):
    d = d[[key, variable]].groupby(key)
    d_m = pd.merge(d.count(), d.max(), left_index=True, right_index=True)
    d_m.rename(columns = {'{}_x'.format(variable): '{}_Count'.format(variable), 
                          '{}_y'.format(variable): '{}_Max'.format(variable)}, inplace = True)
    return d_m.reset_index()
def c(x):
    return x.split(',')
def merge_path(d):
    path_m = d[['CUST_NO', 'new_path']].groupby('CUST_NO')['new_path'].apply(lambda x: "%s" % ','.join(x))
    df_path_m = pd.DataFrame(path_m)
    df_path_m['new_path'] = df_path_m['new_path'].apply(c)
    return df_path_m
# MultiLabelBinarizer
def encoding(d, unique_cust):
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot = MultiLabelBinarizer()
    path_encod = one_hot.fit_transform(d['new_path'])
    c_path = pd.concat([unique_cust, pd.DataFrame(path_encod, columns=one_hot.classes_)], axis=1)
    return c_path

def f1_score_1(str_report):
    return float(list(filter(lambda x:x!='', str_report.split('\n')[3].split(' ')))[3])
def weighted(x):
    NEW_VISITDATE_Max = x[0]
    view = x[1]
    return view * (-30/NEW_VISITDATE_Max)
def scores(tar, met='d', m_size=30):
    train_all, test_all = Create_train_test(31, 61, tar, m_size)
    X_train, Y_train = t_x_y(train_all)
    X_test, Y_test = t_x_y(test_all)
    if met=='d':
        score, ref = decicionTest(X_train, Y_train, X_test, Y_test)
    elif met=='x':
        score, ref = xgboostTest(X_train, Y_train, X_test, Y_test)
    else:
        score, ref = r_d_forestTest(X_train, Y_train, X_test, Y_test)
    return score
def scores(tar, met='d', m_size=30):
    train_all, test_all = Create_train_test(31, 61, tar, m_size)
    
    d = [unsignificant(train_all), unsignificant(test_all)]
    intsection = list(set(d[0]).intersection(*d[:1]))
    
    X_train, Y_train = t_x_y(train_all.drop(intsection, axis=1))
    X_test, Y_test = t_x_y(test_all.drop(intsection, axis=1))
    if met=='d':
        score, ref = decicionTest(X_train, Y_train, X_test, Y_test)
    elif met=='x':
        score, ref = xgboostTest(X_train, Y_train, X_test, Y_test)
    else:
        score, ref = r_d_forestTest(X_train, Y_train, X_test, Y_test)
    return score
def score_test(tar, met):
    for i in [1,2,3,5,6,10,15,30]:
        t1 = time.time()
        print('m_size: {}'.format(i), end=' ')
        score = scores(tar, i, met)
        print('score: {}'.format(score), end=' ')
        t2 = time.time()
        print('time_consumed: {}'.format(t2-t1))

def txn_list(ftd, m_size):
    ltd = ftd + 29
    count = int(30 / m_size)
    k = range(ltd, ltd-30, -m_size)
    return list(zip([ftd]*count, k, [m_size]*count))

# 合併交易(x)及瀏覽
def x_versus_page(x, p, first_txn_day, last_txn_day, first_view_day, last_view_day):
    x_ = selector(x, 'NEW_TXN_DT', first_txn_day, last_txn_day)
    page_ = selector(p, 'NEW_VISITDATE', first_view_day, last_view_day)
    x_page = pd.merge(page_, x_, on='CUST_NO', how='left')[['CUST_NO', 'new_path', 'NEW_VISITDATE', 'NEW_TXN_DT']]
    return x_page

class X(object):
    def __init__(self, first_txn_day, last_txn_day, m_size):
        self.first_txn_day = first_txn_day
        self.last_txn_day = last_txn_day
        self.m_size = m_size
        self.first_view_day = self.last_txn_day - 29 - m_size
        self.last_view_day = self.first_view_day + m_size - 1
    
    def check(self):
        print('{}-{}-{}-{}-{}'.format(self.first_txn_day, self.last_txn_day, 
                        self.m_size, self.first_view_day, self.last_view_day))
        
#     基本資料
    def attach(self):
        self.p=d_page
        self.t=txn
        self.c=card
        self.f=fx
        self.l=loan
        
#     產生屬性
    def generate_1(self, x):
        
        self.x_page = x_versus_page(x, self.p, 
                                    self.first_txn_day, self.last_txn_day, self.first_view_day, self.last_view_day)
        
        self.unique_page = cmm(self.x_page,'CUST_NO', 'NEW_VISITDATE')
        self.unique_txn = cmm(self.x_page,'CUST_NO', 'NEW_TXN_DT')
        self.merge_path = merge_path(self.x_page)
        self.unique_cust = self.x_page['CUST_NO'].drop_duplicates().reset_index().drop('index', axis=1)
        self.c_path = encoding(self.merge_path, self.unique_cust)
        self.all = pd.merge(pd.merge(self.c_path, self.unique_page), self.unique_txn)

def Create_train_test(train_first_txn_day, test_first_txn_day, tar, m_size=30):
    # 建立物件
    trains = [X(a,b,m) for a,b,m in txn_list(train_first_txn_day, m_size)]
    tests = [X(a,b,m) for a,b,m in txn_list(test_first_txn_day, m_size)]
    
    def transform_1(x, tar):
        for i in x:
            i.attach()
        # 選擇目標
        if tar=='t':
            for i in x:
                i.generate_1(i.t)
        elif tar=='c':
            for i in x:
                i.generate_1(i.c)
        elif tar=='f':
            for i in x:
                i.generate_1(i.f)
        elif tar=='l':
            for i in x:
                i.generate_1(i.l)
    transform_1(trains, tar)
    transform_1(tests, tar)
    
    # 篩選交集
    d = [list(i.all.columns) for i in trains+tests]
    intersect = list(set(d[0]).intersection(*d))
    
    def transform_2(x):
        for i in x:
            i.all = i.all[intersect]
        t_all = pd.concat([i.all for i in x])
        t_all['y'] = t_all['NEW_TXN_DT_Count'].apply(lambda x:1 if x>0 else 0)
        t_all.drop(['NEW_TXN_DT_Count', 'NEW_TXN_DT_Max'], axis=1, inplace=True)
        return t_all
    train_all = transform_2(trains)
    test_all = transform_2(tests)
    
    train_all['NEW_VISITDATE_Max'] = train_all['NEW_VISITDATE_Max'].apply(lambda x:x-train_first_txn_day)
    test_all['NEW_VISITDATE_Max'] = test_all['NEW_VISITDATE_Max'].apply(lambda x:x-test_first_txn_day)
    
    def transform_3(x):
        max_group = x[['CUST_NO', 'NEW_VISITDATE_Max', 'y']].groupby('CUST_NO').max().reset_index()
        sum_group = x.drop(['NEW_VISITDATE_Max', 'y'], axis=1).groupby('CUST_NO').sum().reset_index()
        x_all = pd.merge(max_group, sum_group)
        return x_all
    train_all = transform_3(train_all)
    test_all = transform_3(test_all)
    return train_all, test_all

def t_x_y(t_all):
    t_all = pd.merge(t_all, all_identity)
    def x_y(t):
        x = t.drop(['CUST_NO', 'y'], axis=1)
        y = t['y']
        return x, y
    X_t, Y_t = x_y(t_all)
    return X_t, Y_t

def unsignificant(t):
    t_mean = pd.DataFrame(t[t['y']==1].describe().loc['mean'])
    t_mean = t_mean.drop(['NEW_VISITDATE_Max','y', 'NEW_VISITDATE_Count'])
    t_mean = t_mean.sort_values(by='mean', ascending=False)
    return list(t_mean[t_mean['mean']<0.01].index)

# if __name__ == '__main__':
page = pd.read_csv('Downloads/dataset/TBN_CUST_BEHAVIOR.csv')
txn = pd.read_csv('Downloads/dataset/TBN_WM_TXN.csv')
dt = pd.read_csv('Downloads/dataset/TBN_RECENT_DT.csv')
loan = pd.read_csv('Downloads/dataset/TBN_LN_APPLY.csv')
fx = pd.read_csv('Downloads/dataset/TBN_FX_TXN.csv')
card = pd.read_csv('Downloads/dataset/TBN_CC_APPLY.csv')

# 編輯資料
txn['NEW_TXN_DT'] = txn['TXN_DT'].apply(lambda x:x-9447)
card['NEW_TXN_DT'] = card['TXN_DT'].apply(lambda x:x-9447)
fx['NEW_TXN_DT'] = fx['TXN_DT'].apply(lambda x:x-9447)
loan['NEW_TXN_DT'] = loan['TXN_DT'].apply(lambda x:x-9447)

data = pd.read_csv('Downloads/TBN_CUST_BEHAVIOR_PATH.csv')
page['NEW_VISITDATE'] = page['VISITDATE'].apply(lambda x:x-9447)
def add(x):
    path1 = x[0]
    path2 = x[1]
    return '/'.join([x[0],x[1]])
data['new_path'] = data[['PATH1', 'PATH2']].apply(add, axis=1)
d_page = data[['CUST_NO', 'VISITDATE', 'new_path']]
d_page['NEW_VISITDATE'] = d_page['VISITDATE'].apply(lambda x:x-9447)

identity = pd.read_csv('Downloads/dataset/TBN_CIF.csv')
identity.drop('GENDER_CODE', axis=1, inplace=True)
most = {'AGE':3, 'CHILDREN_CNT':0, 'EDU_CODE':3, 'INCOME_RANGE_CODE':1, 'WORK_MTHS':1, 'CUST_START_DT':6260}
all_cust = pd.DataFrame(page['CUST_NO'].drop_duplicates())
all_identity = pd.merge(all_cust, identity, how='left')
all_identity.fillna(value=most, inplace=True)
all_identity['CUST_START_DT'] = all_identity['CUST_START_DT'].apply(lambda x:x-9447)