import numpy as np
import pandas as pd
import datetime
import time
import requests
import astropy.units as u
from astropy.time import Time, TimeDelta
from sunpy.coordinates.sun import carrington_rotation_time
from io import StringIO
import json
from IPython.display import clear_output
import time
import sys
import os

huxt_dir = os.path.join(os.getcwd(), 'HUXt', 'code')
huxt_tools_dir = os.path.join(os.getcwd(), 'HUXt_tools')

sys.path.append(huxt_dir)
sys.path.append(huxt_tools_dir)

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_ensembles as ENS


def huxt_output_to_ml_df(rotation_number, extra_columns, folder_name, overwrite=False, save=False):
    """
    Function to convert the HUXt ensemble output to a usable dataframe for machine learning purposes
    
    args:
    - save            : boolean value - set to "True" to save dataframe to parquet file
    - rotation_number : number of the carrinton rotation
    - extra columns   : array of column names that we may wish to add
                      + options
                         - 'velocity gradient' or 'density'
                         - 'hp30' 
    - folder_name     : name of the folder where HUXt data is stored
    - overwrite       : bool, whether to override current modified data (default=False)
                        
    returns:
    df   : pandas.DataFrame object with all specified variables
    """
    
    # Read in the data for the rotation number
    huxt_data_dir = os.path.join(os.getcwd(), 'src', 'data', 'huxt')

    # Specify load and save locations for the HUXt data
    save_path = os.path.join(huxt_data_dir, f'{folder_name}_modified')
    load_path = os.path.join(huxt_data_dir, folder_name, f'HUXt_rotation_{rotation_number}')
    
    huxt_df = pd.read_parquet(load_path)
    number_ensemble_members = len(huxt_df.columns)

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path) or not os.listdir(save_path) or overwrite:
        os.makedirs(save_path, exist_ok=True)
    else:
        os.makedirs(save_path, exist_ok=False)
    
    # Remove the first 6 days of data (not usable)
    df = huxt_df.drop(huxt_df.index[:6*24])

    # Check whether to include velocity gradient
    if 'velocity gradient' in extra_columns or 'density' in extra_columns:
        # Build new gradient columns all at once to avoid fragmentation
        gradient_cols = {}
    
        for i in range(number_ensemble_members):
            gradient = np.gradient(np.array(df[f'v_{i}']))
            gradient_cols[f'v_{i}_gradient'] = gradient
    
        # Add all new columns at once
        df = pd.concat([df, pd.DataFrame(gradient_cols, index=df.index)], axis=1)
    
        # de-fragment the DataFrame
        df = df.copy()

    # Gather Hpo data for the rotation
    df_cadence = df.index[1] - df.index[0]

    has_hpo = False  # initialise hpo check
    
    # Check if hp30 is asked for and that the df is the correct cadence
    if ('hpo' in extra_columns or 'hp30' in extra_columns) and df_cadence == pd.Timedelta(minutes=30):
        hpo_location = os.path.join(os.getcwd(), 'src', 'data', 'hp30df.parquet')
        hpodf = pd.read_parquet(hpo_location)
        
        df = pd.merge(df, hpodf, left_index=True, right_index=True, how='inner')

    # check for saving
    if save:
        df.to_parquet(os.path.join(save_path, f'HUXt_rotation_{rotation_number}'))
                
    return df


def run_multiple_ambient_ensembles(start_cr, n_crs, n_ensemble, seed, save_folder='', overwrite=False):
    """
    Function to run multiple ambient HUXt ensembles for a specified time
    
    args:
    - start_cr   : which rotation number to start on 
    - n_crs      : number of carrington rotations
    - n_ensemble : number of ensemble members
    - seed       : this fixes the ensemble perturbation parameters

    kwargs:
    - save_folder : name of folder to save the data in 
    - overwrite   : bool - whether to save over previously saved data
                        
    returns:
    - None
    """
    np.random.seed(seed)
    folder_path = os.path.join(os.getcwd(), 'src', 'data', 'HUXt', save_folder)
    
    # Check if the directory exists and is empty
    if not os.path.exists(folder_path) or not os.listdir(folder_path) or overwrite:
        os.makedirs(folder_path, exist_ok=True)
    else:
        # To overwrite a folder, you can change the following 'False' to 'True'
        # Make sure to click run, then change back to 'False' to prevent unwanted overwrites
        os.makedirs(folder_path, exist_ok=False)
    
    rotation_numbers = list(range(start_cr, start_cr + n_crs))
    
    # Set some constants for the model 
    HUXT_TIME_SCALE = 0.5/0.1449375   # magic number used to calibrate the huxt output to hourly output
    SIMTIME = 654 + 6 * 24 + 1
    times = ''
    for cr in rotation_numbers:
        # Get output map
        print(f'Rotation {cr-start_cr+1} / {n_crs}')

        # Get the start time to the nearest hour
        starttime = carrington_rotation_time(cr)
        starttime = starttime.iso[:13]
        starttime = Time(f'{starttime}:00:00.000')

        print(f'CR start time: {starttime}')

        vr_map, lons, lats = Hin.get_MAS_vr_map(cr)

        times, v_in, v_out = ENS.ambient_ensemble(vr_map, lons, lats, 
                                                  starttime=starttime - datetime.timedelta(days=6), 
                                                  simtime=SIMTIME*u.hour,
                                                  dt_scale = HUXT_TIME_SCALE,
                                                  N_ens_amb = n_ensemble)

        # Shift times to the nearest hour to adjust for milliseconds error due to HUXt time step
        adjusted_times = times + TimeDelta(15 * 60, format='sec') 
        times = Time([t[:14] + ('00' if int(t[14:16]) < 30 else '30') + ':00' for t in adjusted_times.iso]) 

        df = pd.DataFrame(data=v_out.T, columns=[f'v_{i}' for i in range(n_ensemble)], index=times.to_datetime())
        save_location = os.path.join(os.getcwd(), 'src', 'data', 'HUXt', save_folder, 'HUXt_rotation_' + str(cr))
        df.to_parquet(save_location)
        clear_output(wait=True)
        

