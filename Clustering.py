#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:25:30 2019

@author: abhishek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import decomposition
from sklearn.preprocessing import scale, minmax_scale
from sklearn import cluster
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean


def clust_func(i, data):
    return cluster.DBSCAN(eps=i, min_samples=5,
                          n_jobs=4).fit_predict(data.values)


def clust_min_func(i, data):
    if i <= 0:
        return np.Inf
    return -len(np.unique(clust_func(i, data)))


def main():
    p_history = '../history_60d.csv'
    df = pd.read_csv(p_history)
    print(df.head())
    
    ntickers0 = len(df['symbol'].unique())
    
    dfx = df.assign(diff=df['adjclose']/df['open'] - 1).pivot_table(
            index='symbol', columns='date', 
            values='diff', aggfunc=np.max).dropna()

    print(dfx.head())
    ntickers1 = len(dfx)
    
    print(f'Number of symbols {ntickers1} ({ntickers1/ntickers0})')
    
    dfscaled = pd.DataFrame(scale(dfx, axis=1), index=dfx.index,
                            columns=dfx.columns)
    
    print(dfscaled.head())
    np.mean(dfscaled.values, axis=1)
    
    res = minimize(fun=clust_min_func, x0=1, args=dfscaled, method='cobyla')
    
    print(f'Optimal eps: {res.x}')
    y = clust_func(res.x, dfscaled)

    print('Clusters:')
    print(np.unique(y))
    
    


if __name__ == '__main__':
    main()