#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:25:53 2019

@author: abhishek
"""
# Standard import

# Third party imports
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def EDA(file, debug=None, tech_analysis=False, start_date=None, end_date=None):
    
    if not file:
        raise "No file provided."

    # Load file
    df = pd.read_csv(file, index_col='Date')
    if debug:    
        print(f"Exploratory data analysis for {file.split('/')[-1]}")

        # Some details on the type of data
        print(df.info())
        print(df.describe())
        print(df.head())
        
        # Check for dataframe containing NaN values
        print(f"Any cell containing NaN value: {df.isnull().values.any()}")
        
        # Plot open and close prices
        ax = plt.gca()
        df.plot(y='Open', ax=ax, grid=True)
        df.plot(y='Close', ax=ax, grid=True)
        plt.show()

    """ Introduce new attributes
        Target is close/open - 1 for regression task and Up_down for 
        classification
    """
    df['Close_close'] = (df['Close']/df['Close'].shift(1)) - 1
    df['High_high'] = (df['High']/df['High'].shift(1)) - 1
    df['Low_low'] = (df['Low']/df['Low'].shift(1)) - 1
    df['Close_open'] = df['Close']/df['Open'] - 1
    df['Up_down'] = np.sign(df['Close_open']).astype('int')
    df['High_low'] = df['High']/df['Low'] - 1
    df['Close_open-1'] = df['Close_open'].shift(1)
    df['Log_volume'] = df['Volume'].apply(np.log)

    if debug:
        # Plot it
        df.plot(y='Close_close')
        plt.show()
        df.plot(y='High_high')
        plt.show()
        df.plot(y='Low_low')
        plt.show()
        df.plot(y='Close_open')
        plt.show()
        df.plot(y='High_low')
        plt.show()    
    
    # Add all Technical analysis features if variable ta is true
    if tech_analysis:
        #df = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume',
        #                            fillna=True)    
        
        df = ta.add_volume_ta(df, 'Open', 'High', 'Low', 'Close', 'Volume')
        
        #df = ta.add_momentum_ta(df, 'Open', 'High', 'Low', 'Close', 'Volume')
        
        df = ta.add_volatility_ta(df, 'Open', 'High', 'Low', 'Close', 'Volume')
        
        df = ta.add_trend_ta(df, 'Open', 'High', 'Low', 'Close', 'Volume')
        
        if debug:        
            print(df.info())
            
            print(f"Starting date = {start_date}")
            print(f"Ending date = {end_date}")
            
            plt.plot(df[start_date:end_date].Close)
            plt.plot(df[start_date:end_date].volatility_bbh, label='High BB')
            plt.plot(df[start_date:end_date].volatility_bbl, label='Low BB')
            plt.plot(df[start_date:end_date].volatility_bbm, label='EMA BB')
            
            plt.title('Bollinger bands')
            plt.legend()
            plt.show()

    return df