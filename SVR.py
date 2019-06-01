#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:23:03 2019

@author: abhishek
"""
import os

from EDA import EDA
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split


quandl.ApiConfig.api_key = "rR9RqufYNmrUGvb-as-G"

INSTRUMENT = 'ETFs/spy.us.txt'

INDICES = ['CME_GC1', 'CBOE_VX1', 'ICE_DX1', 'CME_SP1', 'CME_NG1', 'CME_CL1']
MARKET_FEATURES = ['CHRIS/' + item for item in INDICES]


def standardize(df):
    scaler = preprocessing.StandardScaler()
    columns = df.columns
    
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=columns)
    
    return scaled_df


def support_vector_regression(df, gamma='scale', C=1.0, epsilon=0.2, 
                              features=None, target=None, debug=None):
    X = pd.DataFrame(df[features])    
    y = pd.DataFrame(df[target])
  
    X = standardize(X)
    y = standardize(y)

    if debug:
        print(f"Shape X = {X.shape}, y = {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = SVR(gamma=gamma, C=C, epsilon=epsilon)

    model.fit(X_train, y_train.values.ravel())
    scores = (model.score(X_test, y_test.values.ravel()))
    
    if debug:
        print(f'Scores ({features}, {target}): {scores}')
    else:
        print(f'Score: {scores}')
    
    if X.shape == y.shape:
        plt.scatter(X, y)
        plt.show()


def support_vector_machines(df, gamma='scale', C=1.0, kernel='rbf', 
                            features=None, target=None, debug=None):
    X = pd.DataFrame(df[features])    
    y = pd.DataFrame(df[target])
  
    X = standardize(X)

    if debug:
        print(f"Shape X = {X.shape}, y = {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = SVC(gamma=gamma, C=C, kernel=kernel)

    model.fit(X_train, y_train.values.ravel())
    score = (model.score(X_test, y_test.values.ravel()))
    
    if debug:
        print(f'Score ({features}, {target}): {score}')
    else:
        print(f'Score C={C}: {score}')
    
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


def techical_indicators_svr(df):
    df = df.fillna(method='backfill')
    
    features = list(df.columns)[12:]
    support_vector_regression(df, features=features, target='Close_open')
    support_vector_machines(df, features=features, target='Up_down')    


def index_indicators_svr(df):
    df = add_index_features(df, MARKET_FEATURES, start_date=df.index[0],
                             end_date=df.index[-1])

    df = df.fillna(method='backfill')

    df = df.interpolate(method='linear')
    
    if np.isnan(df.values).any():
        print('NaN values still present, abort...')
        return

    features = list(df.columns)[12:]
    support_vector_regression(df, features=features, target='Close_open')
    support_vector_machines(df, features=features, target='Up_down')
    
    # Test for C
    y = []
    for c in range(1, 100, 1):
        y.append(support_vector_machines(df, C=c, features=features,
                                          target='Up_down'))
    print(f'y = {y}')
    plt.plot(y)
    plt.show()


def main():
    #os.chdir(PROJ_DIR)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    #df = EDA(INSTRUMENT, False, True)
    #techical_indicators_svr(df)

    df = EDA(INSTRUMENT)
    index_indicators_svr(df)


if __name__ == '__main__':
    main()
