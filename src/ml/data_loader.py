# Data loader file

import pandas as pd
import numpy as np
import datetime
import pandas as pd
import os
from sunpy.coordinates.sun import carrington_rotation_time
from astropy.time import Time
import fastparquet

def load_huxt_data_as_windows(data_dir, target, start_cr=1892, end_cr=2278, input_window_size=96, output_window_size=48, post_window_size=24, stride=24, n_ensembles=10, random_sample=True, ensemble_choice=2000):
    """
    This function will split load HUXt data from a specified range of carrington rotation numbers. 
    It will split the data into windows with specified stride and window size. 

    Parameters:
    - data_dir           : os.path - directory where the HUXt data is stored
    - start_cr           : int - starting Carrington Rotation number
    - end_cr             : int - end Carrington Rotation number
    - input_window_size  : int - length of input window in hours (default: 72)
    - output_window_size : int - length of output window in hours (default: 24)
    - post_window_size   : int - length of window in hours after the output window (default: 24)
    - stride             : int - time in hours between start of consecutive windows (default: 24)
    - n_ensembles        : int - number of ensemble members to include
    - random_sample      : bool - determines wethere to take first x ensembles or sample randomly
    - ensemble_choice    : int - specify the number of ensembles in the HUXt run
    
    Returns: 
    - X_windows : array - input variable windows
    - y_windows : array - target variable windows
    - times     : array - times at the start of each window
    """
    
    cols = []
    if random_sample:
        ens = np.random.choice(range(ensemble_choice), n_ensembles, replace=False)
        for i in ens:
            cols.extend([f'v_{i}', f'v_{i}_gradient', f'v_minus_omni_{i}'])
        cols.append(target)
    
    df = pd.read_parquet(os.path.join(data_dir, 'full_df.parquet'), engine='fastparquet', columns=cols)

    X_windows = []
    y_windows = []
    window_size = input_window_size + output_window_size + post_window_size

    starttime = carrington_rotation_time(start_cr)
    starttime = starttime.iso[:13]
    starttime = Time(f'{starttime}:00:00').to_datetime()
    start_row = df.index.get_loc(starttime)
    
    endtime = carrington_rotation_time(end_cr)
    endtime = endtime.iso[:13]
    endtime = Time(f'{endtime}:00:00').to_datetime()
    end_row = df.index.get_loc(endtime)
    
    # Iterate over the DataFrame with the given step size
    for start in range(start_row, end_row - window_size + 1, stride):
        X_window = df.iloc[start:start + window_size, :3*n_ensembles]
        y_window = df[target].iloc[start:start + window_size]
        X_windows.append(X_window)
        y_windows.append(y_window)

    time_windows = np.array([w.index for w in X_windows])
    X_windows, y_windows = np.array(X_windows), np.array(y_windows)
    X_windows = X_windows.reshape(-1, window_size, n_ensembles, 3)
    X_windows = X_windows.transpose((0, 2, 1, 3))

    return X_windows, y_windows, time_windows

def load_omni_data(data_dir):
    """
    Call this to load the OMNI data from any notebook
    """
    df = pd.read_parquet(os.path.join(data_dir, 'OMNI_solar_wind.parquet'))

    # Replace large values with 0 to indicate data gap 
    df = df.where(df <= 9000, 0)
    
    return df

def process_hp30_data():
    """
    Function that gets all of the hpo data
    
    Returns:
    times - dtype = array : datetimes for each hp60 points
    df    - dtype = array : hp60 values
    """
    headers = ('YYYY', 'MM', 'DD', 'hh.h', 'hh._m', 'days', 'days_m', 'Hpo', 'apo', 'D')
   
    filename = 'hpodata.txt'  # Name of text file downloaded from https://kp.gfz.de/en/hp30-hp60/data
    
    df = pd.read_csv(os.path.join(os.getcwd(), 'src', 'data', filename), delimiter='\s+', names=headers)

    df['datetime_str'] = (df['YYYY'].astype(str) + '-' + 
                      df['MM'].astype(str).str.zfill(2) + '-' + 
                      df['DD'].astype(str).str.zfill(2) + ' ' + 
                      df['hh.h'].astype(int).astype(str) + ':' + 
                      (df['hh.h'] % 1 * 60).astype(int).astype(str).str.zfill(2))
    
    # Convert the combined string to datetime
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M')
    
    # Drop the temporary 'datetime_str' column if you don't need it
    df.drop(columns=['datetime_str'], inplace=True)
    times = df['datetime'].to_numpy()

    # Return the array of times, and the Hpo array
    return times, df['Hpo'].to_numpy()