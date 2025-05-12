# Model prediction function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def create_ensemble_models(X_train, X_train_hpo, y_train, model_class, model_hyperparameters):
    """
    Calling this will create models of the specified class and fit them on the training data
    
    Input:
    - X_train     : array - training data of input variables
    - y_train     : array - training data of target variable
    - model_class : class - model Class imported from sklearn / sktime
    - model_hyperparameters : dict - contains the hyperparameters to be passed to the model

    Output:
    - model_array : array - ensemble of fitted models
    """
    
    model_array = []

    if X_train.shape[1] != y_train.shape[0]:
        X_train = np.transpose(X_train, (1, 0, 2))

    for X in X_train:
        if X_train_hpo is not None:
            X = np.hstack((X, X_train_hpo))
        model = model_class(**model_hyperparameters)
        model.fit(X, y_train)
        model_array.append(model)

    return model_array


def make_ensemble_predictions(X_train, X_train_hpo, y_train, model_array, predict_probabilities=False):
    """
    Calling this will train the array of models on the training data

    Input:
    - X_train     : array - training data of input variables
    - y_train     : array - training data of target variable
    - model_array : array - fitted models on the training data (i.e. the ensemble models)

    Output:
    - predictions : array - predictions based on the X and y arrays passed for each of the models
    """
    predictions = []

    if X_train.shape[1] != y_train.shape[0]:
        X_train = np.transpose(X_train, (1, 0, 2))

    for X, model in zip(X_train, model_array):
        if X_train_hpo is not None:
            X = np.hstack((X, X_train_hpo))
        p = model.predict(X)
        predictions.append(p)

    predictions = np.array(predictions).T
    return predictions


def train_final_classifier(X, y, model_type, X_val=None, y_val=None):
    """
    Calling this function will train a model on the given training data with the specified model type

    Input:
    - X          : array - training data of input variables
    - y          : array - training data of target variable
    - model_type : string - type of model to use. 
                    options:
                        - logreg
                        - simple_NN

    Output:
    - model : The trained model
    """

    if model_type in ['logreg', 'logreg_sorted']:
        p, MAE = X
        model = LogisticRegression()
        model.fit(p, y)

    elif model_type[:11] == 'logreg_top_':
        k = int(model_type[11:])
        Nens = X[0].shape[-1]
        n = int((k * Nens) // 100)
        X = np.concatenate([arr[:,:n] for arr in X], axis=-1)
        model = LogisticRegression()
        model.fit(X, y)

    elif model_type == 'persistence':
        model = None

    elif model_type == '27_day_persistence':
        model = None

    elif model_type == 'weighted_mean':
        model = None
    
    else:
        print(f'Model type: {model_type} not recognized')

    return model
    
    
def make_final_classifier_predictions(X, model, model_type, scale=False, storm_thresh=4.66):
    """
    Calling this function will train a model on the given training data with the specified model type

    Input:
    - X          : array - test data of input variables
    - model      : a trained model 
    - model_type : string - type of model as a string
    - scale      : bool - scales probabilities when True

    Output:
    - probabilistic_predictions : np.array - shape = (no. of samples, 1) 
    """
    # Scale data if we aren't using persistence
    if scale and model_type not in ['persistence', '27_day_persistence']: 
        s = StandardScaler()
        s.fit(X[0])
        X[0] = s.transform(X[0])
        
    if model_type == 'logreg' or model_type == 'logreg_sorted':
        p, MAE = X
        probabilistic_predictions = np.expand_dims(model.predict_proba(p)[:, 1], axis=1)

    elif model_type[:11] == 'logreg_top_':
        k = int(model_type[11:])
        Nens = X[0].shape[-1]
        n = int((k * Nens) // 100)
        X = np.concatenate([arr[:,:n] for arr in X], axis=-1)
        probabilistic_predictions = np.expand_dims(model.predict_proba(X)[:, 1], axis=1)

    elif model_type == 'persistence':
        probabilistic_predictions = np.expand_dims(np.max(X, axis=-1) > storm_thresh, axis=-1)

    elif model_type == '27_day_persistence':
        # Define paths
        data_dir = os.path.join(os.path.expanduser('~'), 'storm_forecasting', 'src', 'data')
        huxt_data_dir = os.path.join(data_dir, 'huxt', 'HUXt8_modified')

        # Read in hp30 only
        cols = ['hp30']
        twenty_seven_df = pd.read_parquet(os.path.join(huxt_data_dir, 'full_df.parquet'), engine='fastparquet', columns=cols)

        twenty_seven_df['27_day_offset_hp30'] = twenty_seven_df['hp30'].shift(27 * 24 * 2)
        
        shape = X.shape
        X = X.flatten()
    
        vals = twenty_seven_df['27_day_offset_hp30'].loc[X].to_numpy()
        vals = vals.reshape(shape)
    
        probabilistic_predictions = np.expand_dims(np.max(vals, axis=-1) > storm_thresh, axis=1)
    
    elif model_type == 'weighted_mean':
        p, mae = X
        p = np.clip(p, 0, 1)
        weights = 1 / mae**2
        weights /= np.sum(weights, axis=1, keepdims=True)
        probabilistic_predictions = np.sum(p * weights, axis=1, keepdims=True)

    return probabilistic_predictions

