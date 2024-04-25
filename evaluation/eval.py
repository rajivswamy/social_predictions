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


def generate_binary_prediction_csv(file_path, data_size = 1000, y_true = None, attribute = None):
    """
    Generate synthetic data to unit test eval scripts and stores in user
    provided path

    Parameters:
    file_path (str): user provided path to store generated csv
    data_size (int): number of rows of data to be generated
    """
    # Generate random data for y_true, y_pred, and attribute_gender
    y_pred = np.random.randint(2, size=data_size)

    if (y_true is None) and (attribute is None):
      y_true = np.random.randint(2, size=data_size)
      attribute = np.random.randint(2, size=data_size) # assume binary attribute
      
    # Create a DataFrame using pandas
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'attribute_gender': attribute
    })
    
    # Ensure all parent directories of the given path are made
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)


def batch_generate_binary_prediction_csv(folder_path, num_files = 100, data_size = 1000):
  
  os.makedirs(os.path.dirname(folder_path), exist_ok=True)

  y_true = np.random.randint(2, size=data_size)
  attribute = np.random.randint(2, size=data_size)
  
  for i in range(num_files):
    csv_path = os.path.join(folder_path, f'sample_data{i}.csv')
    generate_binary_prediction_csv(csv_path, data_size, y_true, attribute)

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

    if TP + FP == 0:
        return 0

    return TP/(TP+FP)

def error_rate_ratio(y_true, y_pred):
    TN, FP, FN, TP = _get_conf_mat_values(y_true, y_pred)

    if FP == 0:
        return 0
    
    return FN/FP

def evaluate_model(y_true, y_pred, sensitive_attribute):
    
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
                              'f1_score': f1_score(y_true, y_pred, zero_division=0.0),
                              'precision': precision_score(y_true, y_pred, zero_division=0.0),
                              'recall': recall_score(y_true, y_pred),
                              },
        'fairness_performance': {
                                'by_group_data': mf.by_group.to_dict(), # raw data
                                'difference': mf.difference().to_dict(), # max inter-group diff per stat
                                },
    }

    return results

def evaluate_file(path_obj):
    df = pd.read_csv(path_obj)

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


def batch_evaluate_v2(predictions_path, targets_path, sensitive_attributes_path, write_path = None):
   
    predictions_df = pd.read_csv(predictions_path)
    targets_df = pd.read_csv(targets_path)
    # sensitive_df = pd.read_csv(sensitive_attributes_path)

    random_array = np.random.randint(2, size=targets_df.shape[0])
    sensitive_attribute = pd.Series(random_array, name='sensitive_attribute')

    pred_cols = [col for col in predictions_df if col.startswith('model_')]
    predictions_df = predictions_df[pred_cols]

    per_model_data = {}  
    # use args to get list of prediction vectors, target vector, and sens attr vector
    # Iterate through all the predictions for each model in the rashomon set
    for col in predictions_df:
       y_true = targets_df['Target']
       y_pred = predictions_df[col]

       per_model_data[col] = evaluate_model(y_true, y_pred, sensitive_attribute)

    # Now compute the multiplicity metrics

    mult_data = evaluate_model_multiplicity_v2(predictions_df, sensitive_attribute, groups = [0,1])

    results = {
       "per_model_data": per_model_data,
     "model_multiplicity": mult_data
    }

    # Write data to 'write_name' json file 
    if write_path is not None:
        with open(write_path, "w") as outfile:
            json.dump(results, outfile, indent=4)
    
    return results

def batch_evaluate(folder_path, attr_list, write_name = None):
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
  
  model_mult = evaluate_model_multiplicity(folder_path, attr_list)

  results = {
     "model_performance": perf_data,
     "model_multiplicity": model_mult
  }
  
  # Write data to 'write_name' json file 
  if write_name is not None:
    with open(str(folder/write_name), "w") as outfile:
      json.dump(results, outfile, indent=4)
  
  return results

def load_data_folder(folder_path):
    folder = pathlib.Path(folder_path)
    data_files = list(folder.glob('*.csv'))

    dframes = [pd.read_csv(path_obj) for path_obj in data_files]

    return dframes


def compute_ambiguity(dframes, group = None, attribute_name = None):
    """
    Parameters:
    dframes: list of dataframes
    """

    data = []

    # If group is specified, then limit view to protected group
    if group is not None:
        for df in dframes:
            data.append(df[df[attribute_name]==group])
    else:
        data = dframes
    
    all_preds = np.array([df['y_pred'].to_numpy() for df in data])

    # num_models, num_preds = all_preds.shape --> should hold
    # compute number of unique values for each column
    unique_counts = np.array([len(np.unique(all_preds[:, i])) for i in range(all_preds.shape[1])])

    return (unique_counts > 1).mean()

def compute_ambiguity_v2(preds_df, sensetive_attribute, group = None):
    """
    Parameters:
    dframes: list of dataframes
    """

    data = preds_df.copy()

    # If group is specified, then limit view to protected group
    if group is not None:
        data = preds_df[sensetive_attribute == group]

    all_preds = data.to_numpy().T

    # num_models, num_preds = all_preds.shape --> should hold
    # compute number of unique values for each column
    unique_counts = np.array([len(np.unique(all_preds[:, i])) for i in range(all_preds.shape[1])])

    return (unique_counts > 1).mean()


def compute_discrepancy_v2(preds_df, sensetive_attribute, group = None):
    data = preds_df.copy()

    # If group is specified, then limit view to protected group
    if group is not None:
        data = preds_df[sensetive_attribute == group]

    all_preds = data.to_numpy().T

    num_models, num_preds = all_preds.shape

    max_disc = 0

    # Pass through all model pairings to compute discrepancy
    for i in range(num_models):
        for j in range(i,num_models):
            disagree = (all_preds[i] != all_preds[j]).sum()

            # Change return if needed
            max_disc = max(max_disc, disagree)
    
    return max_disc/num_preds

def compute_discrepancy(dframes, group = None, attribute_name = None):
    data = []

    # If group is specified, then limit view to protected group
    if group is not None:
        for df in dframes:
            data.append(df[df[attribute_name]==group])
    else:
        data = dframes
    
    all_preds = np.array([df['y_pred'].to_numpy() for df in data])
    num_models, num_preds = all_preds.shape

    max_disc = 0

    # Pass through all model pairings to compute discrepancy
    for i in range(num_models):
        for j in range(i,num_models):
            disagree = (all_preds[i] != all_preds[j]).sum()

            # Change return if needed
            max_disc = max(max_disc, disagree)
    
    return max_disc/num_preds

def evaluate_model_multiplicity(folder_path, attr_list):
    dframes = load_data_folder(folder_path)

    results = {}

    # compute total amgig, disc metrics
    results["aggregate"] = {
        "ambiguity": compute_ambiguity(dframes),
        "discrepancy": compute_discrepancy(dframes)
    }

    # compute group level metrics for each attr
    results["attribute"] = {}

    for attr in attr_list:
        results["attribute"][attr] = {}
        for group in [0,1]:
            results["attribute"][attr][group] = {
                "ambiguity": compute_ambiguity(dframes, group, attr),
                "discrepancy": compute_discrepancy(dframes, group, attr)
            }

    return results

def evaluate_model_multiplicity_v2(preds_df, sensetive_attribute, groups = None):

    results = {}

    # compute total amgig, disc metrics
    results["aggregate"] = {
        "ambiguity": compute_ambiguity_v2(preds_df, sensetive_attribute),
        "discrepancy": compute_discrepancy_v2(preds_df, sensetive_attribute)
    }

    # compute group level metrics for each attr
    results["by_group"] = {}

    for group in groups:
        results["by_group"][group] = {
            "ambiguity": compute_ambiguity_v2(preds_df, sensetive_attribute, group),
            "discrepancy": compute_discrepancy_v2(preds_df, sensetive_attribute, group)
            }

    return results