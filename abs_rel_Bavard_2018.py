import scipy.io as sio
import pandas as pd
import numpy as np

# ======================================================================================================================
# Extract the data (Not needed after the first run)
# ======================================================================================================================
# data1 = sio.loadmat('./Data/Magnitude-master/Data_expe1.mat')
# data2 = sio.loadmat('./Data/Magnitude-master/Data_expe2.mat')
#
#
# def extract_data(data, training_var, transfer_var):
#
#     # Define column name per variable
#     name_dict = {'cho': 'choice', 'con': 'trial_all', 'con2': 'trial', 'con3': 'partial_feedback', 'out': 'outcome',
#                  'out2': 'outcome_rel', 'cou': 'unchosen_outcome', 'cou2': 'unchosen_outcome_rel',
#                  'ss': 'transfer_choice', 'aa': 'transfer_trial', 'subjects': 'sub', 'phase': 'phase'}
#
#     # Extract the data
#     df = pd.DataFrame()
#
#     # Iterate over the items in the mat_data dictionary
#     for key, value in data.items():
#         # Ignore metadata keys and check if it's a matrix or array-like structure
#         if not key.startswith('__') and isinstance(value, (np.ndarray, list)):
#             # Flatten the matrix (if 2D or higher) into 1D and assign to a column
#             flat_value = np.ravel(value)  # Flatten the matrix into 1D array
#             df[key] = pd.Series(flat_value)
#
#     # Unnest the data
#     training_data = df.explode(training_var).drop(columns=transfer_var)
#     transfer_data = df.explode(transfer_var).drop(columns=training_var)
#
#     for data in [training_data, transfer_data]:
#         for col in data.columns:
#             data[col] = data[col].apply(lambda x: x[0] if hasattr(x, '__getitem__') and len(x) == 1 else x)
#
#     # combine the data
#     training_data['phase'] = 'training'
#     transfer_data['phase'] = 'transfer'
#     df = pd.concat([training_data, transfer_data]).reset_index(drop=True)
#
#     # rename the columns
#     df.columns = [name_dict[col] for col in df.columns]
#
#     return df
#
#
# data1 = extract_data(data1, ['cho', 'con', 'con2', 'out', 'out2'], ['ss', 'aa'])
# data2 = extract_data(data2, ['cho', 'con', 'con2', 'con3', 'out', 'out2', 'cou', 'cou2'], ['ss', 'aa'])
#
# data1.to_csv('./Data/Magnitude-master/Data_expe1.csv', index=False)
# data2.to_csv('./Data/Magnitude-master/Data_expe2.csv', index=False)

# ======================================================================================================================
# Load the data
# ======================================================================================================================
data1 = pd.read_csv('./Data/Magnitude-master/Data_expe1.csv')
data2 = pd.read_csv('./Data/Magnitude-master/Data_expe2.csv')