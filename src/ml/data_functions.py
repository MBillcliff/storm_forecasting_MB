# Functions for organising data

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_val_split(X_windows, y_windows, times, seed):
    print('Splitting into training, testing, and validation...')
    n_windows = len(X_windows)
    indices = np.arange(n_windows)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices = indices[:int(n_windows * 0.7)]
    val_indices = indices[int(n_windows*0.7):int(n_windows*0.85)]
    test_indices = indices[int(n_windows*0.85):]

    X_train, y_train, times_train = X_windows[train_indices], y_windows[train_indices], times[train_indices]
    X_val, y_val, times_val = X_windows[val_indices], y_windows[val_indices], times[val_indices]
    X_test, y_test, times_test = X_windows[test_indices], y_windows[test_indices], times[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test, times_train, times_val, times_test


def balance_data(X_windows, y_windows, times, input_window_size, post_window_size, storm_threshold=4.66):
    """
    Call this to balance the number of storms and non-storms 

    Parameters:
    - X_windows : array - input data
    - y_windows : array - target variables
    - times     : array - corresponding times for input data
    - input_window_size : int - hours preceding forecast window
    - post_window_size   : int - hours after the output window

    Returns: 
    - X_balanced     : array - input data reduced to include balanced storm and non-storm
    - y_balanced     : array - target variable reduced to include balanced storm and non-storm
    - times_balanced : array - balanced array of times

    NB: Only works when: no. storms >= no. non-storms
    """
    
    print('Balancing...')
    
    # get the max hpo value for each output window
    y_max_hpo = np.max(y_windows[:, input_window_size:-post_window_size], axis=1)

    # Extract indices for when we exceed large storm threshold and when we don't exceed storm threshold
    storm_indices = np.where(y_max_hpo >= storm_threshold)[0]
    non_storm_indices = np.where(y_max_hpo < storm_threshold)[0]
    
    # Randomly drop non-storms to balance with the storm times
    non_storm_indices = np.random.choice(non_storm_indices, size=len(storm_indices), replace=False)

    # Combine indices
    all_indices = np.concatenate((storm_indices, non_storm_indices))

    # Extract correct parts of our arrays
    X_balanced = X_windows[all_indices]
    y_balanced = y_windows[all_indices]
    times_balanced = times[all_indices]

    return X_balanced, y_balanced, times_balanced


def convert_hpo_to_boolean(y_windows, input_window_size, post_window_size, storm_threshold=4.66):
    """
    Call this to convert array of hpo values to binary classification based on specified storm_threshold

    Parameters:
    - y_windows : array - target variables
    - input_window_size : int - time steps preceding forecast window
    - post_window_size  : int - time steps after the forecast window
    - storm_threshold   : float - threshold to split data on

    Returns: 
    - y_windows : same as input array
    """
    
    y_windows = np.max(y_windows[:, input_window_size:-post_window_size], axis=1) >= storm_threshold
    
    return np.array(y_windows)


# def get_storms(X, y, times, threshold=4.66, input_window_size=24, post_window_size=24):
#     """
#     Call this function to remove all data where max value within output window is less than the specified threshold

#     Parameters:
#     - X : array - input data
#     - y : array - target variables as hp60 values
#     - times     : array - corresponding times for input data
#     - input_window_size : int - hours in input window
#     - post_window_size   : int - hours after the output window

#     Returns: 
#     - X_storm     : array - input data reduced to include balanced storm and non-storm
#     - y_storm     : array - target variable reduced to include balanced storm and non-storm
#     - times_storm : array - balanced array of times
#     """

#     # Gather indices at which we meet the threshold
#     print(y.shape)
#     storm_indices = np.any(y > threshold, axis=1)

#     X_storm, y_storm, times_storm = X[storm_indices], y[storm_indices], times[storm_indices]

#     return X_storm, y_storm, times_storm
    

    
    








    