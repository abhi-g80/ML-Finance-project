#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 01:36:17 2019

@author: abhishek
"""
from pathlib import Path

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range = (0, 1))

INSTRUMENT = 'ETFs/spy.us.txt'


class CNNModel():
    """ Wrapper class for implementing CNN model for predicting future price
    movement.
    
    Args:
    -----
        instrument:     OHLC file for the instument to be analyzed.
        split:          Percentage of train and test split, default 80%.
        look_back_days: Number of days to be used by CNN, default 60 days.
        target:         Feature to predict, default 'Close'.
        epochs:         Number of epochs for CNN, default 5.
        batch_size:     Batch size for CNN, default 32.
    """

    def __init__(self, instrument=None, split=0.8, look_back_days=60, 
                 target='Close', epochs=5, batch_size=32,
                 savegraph='graph.png'):
        self.instrument = instrument
        self.split = split
        self.target = target
        self.look_back_days = look_back_days
        self.batch_size = batch_size
        self.epochs = epochs
        self.savegraph = savegraph
        self.model = None
        self.train = None
        self.test = None
        self.predictions = None
        self.__instr_df = None
        self.__actual_prices = None
        self.__actual_test_prices = None

    def __repr__(self):
        return self.__dict__
    
    def __str__(self):
        return f"""CNNModel(instrument={self.instrument},
        split={self.split}, target={self.target},
        look_back_days={self.look_back_days})"""

    @property
    def instrument(self):
        return self.__instrument

    @property
    def split(self):
        return self.__split

    @property
    def look_back_days(self):
        return self.__look_back_days
    
    @property
    def epochs(self):
        return self.__epochs
    
    @property
    def batch_size(self):
        return self.__batch_size

    @instrument.setter
    def instrument(self, instrument):
        if instrument:
            if Path(instrument).is_file():
                self.__instrument = instrument
            else:
                raise FileNotFoundError(f"{instrument} doesn't exist.")

    @split.setter
    def split(self, split):
        if 0.5 < split < 1:
            self.__split = split
        else:
            raise ValueError(f"Split ratio given {split}, should be between "
                              "0.50 and 0.99.")
    
    @look_back_days.setter
    def look_back_days(self, look_back_days):
        if look_back_days > 20:
            self.__look_back_days = look_back_days
        else:
            raise ValueError(f"Look back days given {look_back_days}, has to "
                              "be greater than 20")
    
    @epochs.setter
    def epochs(self, epochs):
        if 1 <= epochs <= 200:
            self.__epochs = epochs
        else:
            raise ValueError(f"Epochs given {epochs}, should be between "
                              "1 and 100")
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if 32 <= batch_size <= 512:
            self.__batch_size = batch_size
        else:
            raise ValueError(f"Batch size given {batch_size}, should be "
                              "between 32 and 512")

    def __process_data(self):
        self.__instr_df.interpolate(method='linear')

    def __split_train_and_test_df(self):
        self.__process_data()

        index = self.__instr_df.columns.get_loc(self.target)
        nparray = self.__instr_df.iloc[:,index:(index+1)]

        self.__actual_prices = nparray

        # Normalize our Numpy array
        nparray = self.normalize(nparray)

        # Train test split, default 80% train 20% test
        total_obs = len(nparray)
        self.train = nparray[:int(total_obs * self.split),:]

        test_inputs = nparray[len(self.train) - self.look_back_days:]

        test_inputs = test_inputs.reshape(-1,1)

        test_features = []
        for i in range(self.look_back_days, len(test_inputs)):
            test_features.append(test_inputs[i - self.look_back_days:i,0])

        test_features = np.array(test_features)
        test_features = np.reshape(test_features, (test_features.shape[0],
                                                   test_features.shape[1], 1))

        self.test = test_features
        return

    def __plot(self):
        fig, ax = plt.subplots(figsize=(8,5))
        self.__actual_test_prices.plot(color='blue', grid=True, ax=ax)
        ax.plot(self.predictions , color='red', label='Predicted close')
        ax.set_title('Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        plt.savefig(self.savegraph)
        plt.show()
        return
    
    def __read(self):
        if self.instrument:
            self.__instr_df = pd.read_csv(self.instrument, index_col='Date')

    def preprocess(self):
        self.__read()
        self.__split_train_and_test_df()
        return

    def normalize(self, df, inverse=False):
        if inverse:
            df_scaled = scaler.inverse_transform(df) 
        else:
            df_scaled = scaler.fit_transform(df)
        return df_scaled

    def plot(self):
        self.predictions = self.normalize(self.predictions, inverse=True)
        self.__actual_test_prices = self.__actual_prices[len(self.train):]
    
        if self.__actual_test_prices.shape == self.predictions.shape:
            self.__plot()
        else:
            print("Shape doesn't match")
            print(f"Actual price shape {actual_prices.shape}")
            print(f"Prediction shape {predictions.shape}")

        return

    def fit(self, optimizer='adam', loss='mean_squared_error'):
        features_set = []
        labels = []

        for i in range(self.look_back_days, len(self.train)):
            features_set.append(self.train[i-self.look_back_days:i, 0])
            labels.append(self.train[i, 0])

        features_set, labels = np.array(features_set), np.array(labels)

        features_set = np.reshape(features_set, (features_set.shape[0], 
                                             features_set.shape[1], 1))
        
        print(features_set.shape)

        self.model = Sequential()
        self.model.add(Conv1D(64, (4,), activation='relu', 
                       input_shape=(features_set.shape[1], 1)))   

        self.model.add(MaxPooling1D(pool_size=(2,)))
        self.model.add(Conv1D(64, (4,), activation='relu'))
        
        #self.model.add(MaxPooling1D(pool_size=(2,)))
        #self.model.add(Conv1D(64, (4,), activation='relu'))        

        self.model.add(MaxPooling1D(pool_size=(2,)))
        self.model.add(Flatten())
        
        self.model.add(Dense(64, activation='relu'))
        
        self.model.add(Dense(1, activation='sigmoid'))
    
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=['mae', 'acc'])
        self.history = self.model.fit(features_set, labels,
                                      epochs=self.epochs, 
                                      batch_size=self.batch_size)

    def predict(self):
        self.predictions = self.model.predict(self.test)
        return

    def error(self):
        """ Naive RMSE implementation """
        return ((self.__actual_test_prices[self.target].values - self.predictions)
                ** 2).mean() ** 0.5
        

def main():
    # Create instance
    instr = CNNModel(INSTRUMENT, split=0.8, look_back_days=60, epochs=100,
                     savegraph='Graphs/CNN_e100_2layer.png')
    
    # Read and process the data
    instr.preprocess()
    
    # Fit CNN model
    instr.fit()
    
    # Predict
    instr.predict()
    
    # Plot actual vs prediction
    instr.plot()
    
    print(f'RMS error: {instr.error()}')

if __name__ == '__main__':
    main()
