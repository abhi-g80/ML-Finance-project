#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:29:30 2019

@author: abhishek
"""
import os

import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EDA import EDA

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score

from lightgbm import LGBMClassifier


quandl.ApiConfig.api_key = "rR9RqufYNmrUGvb-as-G"

INSTRUMENT = 'ETFs/spy.us.txt'

INDICES = ['CME_GC1', 'CBOE_VX1', 'ICE_DX1', 'CME_SP1', 'CME_NG1', 'CME_CL1']
MARKET_FEATURES = ['CHRIS/' + item for item in INDICES]


scaler = preprocessing.StandardScaler()


def custom_train_test_split(X, y, test_size=0.2):
    X_train, X_test = X[:int(len(X)*(1-test_size))], X[int(len(X)*(1-test_size)):]
    y_train, y_test = y[:int(len(y)*(1-test_size))], y[int(len(y)*(1-test_size)):]
    
    return X_train, X_test, y_train, y_test


def standardize(df):
    columns = df.columns
    
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=columns)
    
    return scaled_df


def lightgbm_classifier(df, features=None, target=None, debug=None):
    X = pd.DataFrame(df[features])    
    y = pd.DataFrame(df[target])
  
    X = standardize(X)

    if debug:
        print(f"Shape X = {X.shape}, y = {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False)

    params = {
        'objective':'multiclass', 
        'boosting':'gbdt', 
        'metric':'multi_logloss', 
        'num_boost_round':50000, 
        'random_state':5,
        'reg_lambda': 1.2,
        'reg_alpha': 1,
    }
    
    lgbm = LGBMClassifier(**params)
    
    model = lgbm.fit(X_train, y_train)
    
    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)
    
    print(f'Score: {score}')
    
    return


def logistic_regression(df, features=None, target=None, debug=None):
    X = pd.DataFrame(df[features])    
    y = pd.DataFrame(df[target])
  
    X = standardize(X)

    if debug:
        print(f"Shape X = {X.shape}, y = {y.shape}")

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y,
                                                               test_size=0.2)
    model = LogisticRegression(random_state=0, solver='lbfgs',
                               multi_class='multinomial')

    model.fit(X_train, y_train.values.ravel())
    score = (model.score(X_test, y_test.values.ravel()))
    
    if debug:
        print(f'Score ({features}, {target}): {score}')
    else:
        print(f'Score: {score}')
    
    if X.shape == y.shape:
        plt.scatter(X, y)
        plt.show()
    
    return score    


def random_forest_classifier(df, features=None, n_estimators=100, max_depth=2,
                             target=None, debug=None):
    X = pd.DataFrame(df[features])    
    y = pd.DataFrame(df[target])
  
    X = standardize(X)

    if debug:
        print(f"Shape X = {X.shape}, y = {y.shape}")

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y,
                                                               test_size=0.2)
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                   max_depth=max_depth, random_state=0)

    model.fit(X_train, y_train.values.ravel())
    score = (model.score(X_test, y_test.values.ravel()))
    
    if debug:
        print(f'Score ({features}, {target}): {score}')
    else:
        print(f'Score: {score}')
    
    if X.shape == y.shape:
        plt.scatter(X, y)
        plt.show()
    
    return score


def support_vector_machines(df, gamma=0.025, C=55.0, kernel='rbf', 
                            features=None, target=None, debug=None):
    X = pd.DataFrame(df[features])    
    y = pd.DataFrame(df[target])
  
    X = standardize(X)

    if debug:
        print(f"Shape X = {X.shape}, y = {y.shape}")

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y,
                                                               test_size=0.2)
    model = SVC(gamma=gamma, C=C, kernel=kernel)

    model.fit(X_train, y_train.values.ravel())
    score = (model.score(X_test, y_test.values.ravel()))
    
    if debug:
        print(f'Score ({features}, {target}): {score}')
    else:
        print(f'Score: {score}')
    
    if X.shape == y.shape:
        plt.scatter(X, y)
        plt.show()
    
    return score



def add_index_features(df, args, start_date=None, end_date=None):
    if start_date and end_date:
        for arg in args:
            try:
                feature = quandl.get(arg, start_date=start_date, 
                                     end_date=end_date)
                feature.rename(columns={name:arg + ' ' + name 
                                        for name in feature.columns},
                               inplace=True)
                df = df.join(feature)
            except Exception as e:
                print(f'{e}')
                return
    else:
        print('No start and end date provided')
        return

    return df


def techical_indicators(df):
    df = df.fillna(method='backfill')
    
    features = list(df.columns)[11:]
    
    print('Logistic regression - ', end='')
    logistic_regression(df, features=features, target='Up_down')
    
    print('Support vector machine - ', end='')
    support_vector_machines(df, features=features, target='Up_down')
    
    print('Random forest classifier - ', end='')
    random_forest_classifier(df, features=features, n_estimators=100,
                             target='Up_down')    

    print('LightGBM classifier - ', end='')
    lightgbm_classifier(df, features=features, target='Up_down')


def index_indicators(df):
    df = add_index_features(df, MARKET_FEATURES, start_date=df.index[0],
                             end_date=df.index[-1])

    df = df.fillna(method='backfill')

    df = df.interpolate(method='linear')

    if np.isnan(df.values).any():
        print('NaN values still present, abort...')
        return

    features = list(df.columns)[11:]
    
    print('Logistic regression - ', end='')
    logistic_regression(df, features=features, target='Up_down')
 
    print('Support vector machine - ', end='')
    support_vector_machines(df, features=features, target='Up_down')    
    
    print('Random forest classifier - ', end='')
    random_forest_classifier(df, features=features, n_estimators=100,
                             target='Up_down')
    
    print('LightGBM classifier - ', end='')
    lightgbm_classifier(df, features=features, target='Up_down')


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    df = EDA(INSTRUMENT, False, True)
    df = df.drop('Close_open', axis=1)
    print(df.columns)
    print("Techincal indicators")
    techical_indicators(df)

    print()
    
    df = EDA(INSTRUMENT)
    df = df.drop('Close_open', axis=1)
    print(df.columns)
    print("Market indicators")
    index_indicators(df)


if __name__ == '__main__':
    main()
