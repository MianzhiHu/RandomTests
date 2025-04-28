import numpy as np
import pandas as pd
from utils.ComputationalModeling import ComputationalModels, dict_generator

# ====================================================================
# Load the data
# ====================================================================
data = pd.read_csv('./Data/WSLL1.csv')

# ====================================================================
# Preprocess the data
# ====================================================================
# Convert letters to numbers
data['SGTBinChoice'] = data['SGTBinChoice'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
data_dict = dict_generator(data, task='IGT_SGT')

test_data = data[data['ID'] == 1]
test_dict = dict_generator(test_data, task='IGT_SGT')
# ====================================================================
# Define the model
# ====================================================================
delta_PVL = ComputationalModels('delta_PVL', task='IGT_SGT', condition='Both', num_trials=100)

if __name__ == '__main__':
    # ====================================================================
    # Fit the model
    # ====================================================================
    result = delta_PVL.fit(data_dict, num_iterations=1)