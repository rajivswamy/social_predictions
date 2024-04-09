# General Imports
import numpy as np
import pandas as pd
import os
import pathlib
import json

# Evaluation and Fairness imports
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate, # false positive error rate balance
    true_positive_rate, # false negative error rate balance
)

from sklearn.metrics import ( 
    accuracy_score, # use for both performance and fairness parts
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def generate_binary_prediction_csv(file_path, data_size = 1000):
    """
    Generate synthetic data to unit test eval scripts and stores in user
    provided path

    Parameters:
    file_path (str): user provided path to store generated csv
    data_size (int): number of rows of data to be generated
    """
    # Generate random data for y_true, y_pred, and attribute_gender
    y_true = np.random.randint(2, size=data_size)
    y_pred = np.random.randint(2, size=data_size)
    attribute_gender = np.random.randint(2, size=data_size) # assume binary attribute
    
    # Create a DataFrame using pandas
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'attribute_gender': attribute_gender
    })
    
    # Ensure all parent directories of the given path are made
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

# Quick utility function to get confusion matrix
def _get_conf_mat_values(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)

  # Extracting confusion matrix values
  TN = cm[0, 0]
  FP = cm[0, 1]
  FN = cm[1, 0]
  TP = cm[1, 1]

  return TN, FP, FN, TP

def predictive_parity(y_true, y_pred):
  TN, FP, FN, TP = _get_conf_mat_values(y_true, y_pred)
  return TP/(TP+FP)

def error_rate_ratio(y_true, y_pred):
  TN, FP, FN, TP = _get_conf_mat_values(y_true, y_pred)
  return FN/FP

def evaluate_file(path_obj):
    df = pd.read_csv(path_obj)

    model_name = path_obj.name

    y_true = df['y_true']
    y_pred = df['y_pred']
    sensitive_attribute = df['attribute_gender']
    
    # Fairness measurements processing
    metrics = {
              'selection_rate': selection_rate,
              'ppv': predictive_parity,
              'fp_err_rate_balance': false_positive_rate,
              'tp_error_rate_balance': true_positive_rate,
              'accuracy': accuracy_score,
              'error_rate_ratio': error_rate_ratio,
               }

    mf = MetricFrame(
                    metrics=metrics,
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_attribute
                    )
    
    results = {
        'model_performance': {'accuracy': accuracy_score(y_true, y_pred),
                              'f1_score': f1_score(y_true, y_pred),
                              'precision': precision_score(y_true, y_pred),
                              'recall': recall_score(y_true, y_pred),
                              },
        'fairness_performance': {
            'by_group_data': mf.by_group.to_dict(), # raw data
            'difference': mf.difference().to_dict(), # max inter-group diff per stat
            },
    }

    return results

def batch_evaluate(folder_path, write_name = None):
  """
  Processes model result CSV files in the 

  Parameters:
  folder_path (str): user provided path to folder
  write_name (str): name of json file for output to be writted, 
                    output is written to json file in same folder

  Returns
  list of dicts containing performance results
  """
  folder = pathlib.Path(folder_path)

  perf_data = {}

  # Iterate through all the csv files in the folder
  for path in list(folder.glob('*.csv')):
    perf_data[str(path)] = evaluate_file(path)
  
  # Write data to 'write_name' json file 
  if write_name is not None:
    with open(str(folder/write_name), "w") as outfile:
      json.dump(perf_data, outfile, indent=4)
  
  return perf_data