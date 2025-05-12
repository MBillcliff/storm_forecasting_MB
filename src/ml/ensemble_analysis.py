#Ensemble analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.colors import Normalize
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import os

fig_dir = os.path.join(os.getcwd(), 'src', 'figures')

def evaluate_predictions(y_pred, y_true, threshold=0.5):
    """
    Evaluate predictions using various classification metrics.

    Parameters:
    - y_pred: array, predicted probabilities
    - y_true: array, true labels

    Optional Parameters
    - threshold: float, threshold for classification (default: 0.5)

    Returns:
    - metrics_dict: dictionary, containing classification metrics
    """
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten().astype(int)
    
    # Ensure the number of rows in predictions matches the length of true_labels
    if y_pred.shape != y_true.shape:
        raise ValueError("Mismatch in the number of rows between predictions and true_labels.")

    # Iterate through ensemble members          
    y_pred_labels = (y_pred > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    # ROC Curve
    fpr, tpr, thresh = roc_curve(y_true.flatten(), y_pred.flatten())
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix for TP, TN, FP, FN
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred_labels).ravel()

    # Peirce Skill Score
    peirce_ss = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

    # Pearson Skill Score
    pearson_ss = (TP + TN) / (TP + FP + TN + FN)

    # Heidke Skill Score
    heidke_ss = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

    # Calculate average metrics across ensemble members
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Peirce Skill Score': peirce_ss,
        'Pearson Skill Score': pearson_ss,
        'Heidke Skill Score': heidke_ss,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
    }

    return metrics_dict

def metric_table_plot(df, metric, x_labels, y_labels, model_name='', x_axis_name='', y_axis_name='', save=False, tag='', run_number=0):
    """
    Plots the metric values for a specified metric in a grid 
    
    Parameters:
    - x_labels: array, labels that will go along the x_axis
    - y_labels: array, labels that will go along the y_axis
    - metric: string, must match a header in the metric pandas.DataFrame

    Optional Parameters: 
    - x_axis_name: string, label for x_axis
    - y_axis_name: string, label for y_axis
    - save: bool, set to True to save the plot
    - tag : str, use this to overwrite the default save name

    Returns:
    - None
    """
    fig_dir = os.path.join(os.path.expanduser('~'), 'HUXt', 'figures')

    data = np.array(df[metric])
    data = data.reshape(len(x_labels), len(y_labels))

    # Create a 4x4 plot with associated numbers and colors
    plt.imshow(data, cmap='viridis')

    # Add number annotations
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white')

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label(metric)

    # Add tick labels
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.yticks(np.arange(len(y_labels)), y_labels)

    # Set axis labels
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    if save:
        if tag == '':
            print('NOT SAVING QUADRANT PLOT - NO TAG SPECIFIED')
        else: 
            save_loc = os.path.join(fig_dir, 'metric_table_plots', f'{tag}.png')
            plt.savefig(save_loc, bbox_inches='tight')
    
    # Show plot
    plt.title(f'Window Size Effect on {model_name} Performance')
    plt.show()


def single_plot(predicted, X, y, y_hpo, times, ensemble_predictions, maes, buffer, input_window_size, output_window_size, post_window_size, run_number, OMNI=None, save=False, subfolder='', tag='',):
    """    
    Parameters:
    - probabilistic_predictions: array, predicted probabilities
    - times: array, times for the corresponding X values
    - X: array, input data
    - y: array, target values in the form of storm (1) or non-storm (0)
    - y_hpo: array, target values in the form of hpo values
    - ensemble_predictions: array, predictions from each huxt ensemble predictor
    - maes : array, associated MAE for each huxt output with OMNI
    - input_window_size: array, number of hours before prediction window
    - output_window_size: array, number of hours within the prediction window
    - post_window_size: array, number of hours after the prediction window

    Optional Parameters: 
    - OMNI: pandas.DataFrame, Solar wind velocities from OMNI to plot over ensemble members (default: None)
    - save: bool, set to True to save the plot (default: False)
    - tag : str, use this to overwrite th e default save name (default: '')
    - cmap_name: str, colour map to use for the velocities (default: None, will use blue/red and an alpha)

    Returns:
    - None
    """
    
    fig_dir = os.path.join(os.getcwd(), 'src', 'figures')

    # Setup axes
    fig, (ax2, ax3,)  = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

    labelsize=14
    # ax1.set_ylabel('Velocity (km/s)', fontsize=labelsize)
    ax2.set_ylabel('Velocity (km/s)', fontsize=labelsize)
    ax3.set_ylabel('Hp30', fontsize=labelsize)
    ax3.set_xlabel('Time', fontsize=labelsize)

    max_hpo = max(y_hpo[input_window_size:-post_window_size])
    ax2.set_title(f'Model Probability = {predicted:.3f}    Max Hpo = {max_hpo}')

    if (predicted >= 0.5) == y:
        # ax1.set_facecolor((0.9, 1, 0.9))
        ax2.set_facecolor((0.9, 1, 0.9))
        ax3.set_facecolor((0.9, 1, 0.9))
    else: 
        # ax1.set_facecolor((1, 0.9, 0.9))
        ax2.set_facecolor((1, 0.9, 0.9))
        ax3.set_facecolor((1, 0.9, 0.9))
        
    ax3.tick_params(axis='x', labelrotation=25, grid_snap=1, labelsize=12)
    date_format = DateFormatter('%Y-%m-%d %H')
    ax3.xaxis.set_major_formatter(date_format)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(8))

    ax3.axhline(4.66, color='gray', linestyle='--', label='Storm threshold')
    ax3.set_xlim(times[0], times[-1])
            
    for ax in (ax2, ax3):
        pred_label = "Forecast Window" if ax == ax2 else None
        buff_label = "Buffer Region" if ax == ax2 else None
        ax.axvspan(times[input_window_size], 
               times[input_window_size + output_window_size - 1],
               color='gray', alpha=0.4, label=pred_label)
        ax.axvspan(times[input_window_size - buffer], 
                       times[input_window_size],
                       color='gray', alpha=0.2, label=buff_label)

    # Set the alpha for each line in the velocity plots
    alpha = 1

    has_storm_label = False
    has_non_storm_label = False
    X = X.squeeze()

    # get hourly times and the corresponding OMNI data
    if OMNI is not None:
        try:
            t = times[::2]
            omni_velocities = OMNI.loc[t.flatten()]
            omni_velocities[omni_velocities == 0] = np.nan
            color = 'k'
        except:
            t = times[1::2]
            omni_velocities = OMNI.loc[t.flatten()]
            omni_velocities[omni_velocities == 0] = np.nan
            color = 'k'

    for i in range(len(X)):
        cmap1 = plt.get_cmap('coolwarm')
        prediction = ensemble_predictions.squeeze()[i]
        color = cmap1(prediction)  # Map prediction value to a color from the colormap
        # ax1.plot(times, X[i].T[0], color=color, alpha=alpha)
        
        low, high = 0, 200
        cmap2 = plt.get_cmap('Blues')

        v = X[i].T[0][:input_window_size-buffer:2]

        v_minus_omni = np.abs(omni_velocities[:(input_window_size - buffer)//2].squeeze() - v)
        v_minus_omni = v_minus_omni[~np.isnan(v_minus_omni)]
        mae = np.mean(v_minus_omni)
        
        norm = Normalize(vmin=low, vmax=high)  # Normalize MAE to [0, 1]
        color = cmap2(norm(mae))
        ax2.plot(times, X[i].T[0], color=color)

    # Include colour bars to the side of each of the panels
    # cbar_ax1 = fig.add_axes([0.88, 0.69, 0.02, 0.25])  # [left, bottom, width, height]
    # cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap1), cax=cbar_ax1)
    # cbar1.set_label('Storm Probability')
    
    cbar_ax2 = fig.add_axes([0.88, 0.55, 0.02, 0.4])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap2), cax=cbar_ax2)
    cbar2.set_label('MAE (km/s)')
    
    # plot OMNI
    # ax1.plot(t, omni_velocities, color='k')
    ax2.plot(t[:(input_window_size - buffer)//2 + 1], omni_velocities[:(input_window_size - buffer)//2 + 1], color='k', label='OMNI')
    ax2.plot(t[(input_window_size - buffer)//2:], omni_velocities[(input_window_size - buffer)//2:], color='k', label=None)

    # Plot Hpo data 
    ax3.plot(times[:input_window_size - buffer + 1], y_hpo.squeeze()[:input_window_size - buffer + 1], label='Hp30', color='#1f77b4')
    ax3.plot(times[input_window_size - buffer:], y_hpo.squeeze()[input_window_size - buffer:], color='#1f77b4', label=None)
    
    # Create legend
    legend_loc = (0.85, 0.3)
    legend = fig.legend(loc='center left', bbox_to_anchor=legend_loc, fontsize=10)

    # adjust spacing for panels
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1, hspace=0.1)

    # Save everything to the corresponding folder
    if save:
        if tag == '' or subfolder == '':
            print('NOT SAVING QUADRANT PLOT - NO LOCATION SPECIFIED')
        else:
            save_loc = os.path.join(fig_dir, 'single_plots', subfolder, f'{tag}.png')
            plt.savefig(save_loc, bbox_inches='tight')
    plt.show()
    

    