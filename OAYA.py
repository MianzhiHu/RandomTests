import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from utils.ComputationalModeling import ComputationalModels, dict_generator, parameter_extractor
from utils.DualProcess import DualProcessModel

# ====================================================================
# Load the data
# ====================================================================
data1 = pd.read_csv('./Data/CombOAYA_SCGTData.csv')
data2 = pd.read_csv('./Data/SGTOAYAData_2023.csv', dtype={9: str})

# ====================================================================
# Preprocess the data
# ====================================================================
# Data 1
data1['choice'] = data1['choice'] - 1
data1['AgeGroup'] = np.where(data1['subjID'].between(300, 399), 'OA', 'YA')
data1['subjID'] = data1['subjID'] - 300
data1_dict = dict_generator(data1, task='IGT_SGT')

# Data 2
data2['choice'] = data2['choice'] - 1
data2['trial'] = data2.groupby('subject').cumcount() + 1
data2 = data2[data2['trial'] <= 100].reset_index(drop=True)
data2['Subnum'] = (data2.index // 100) + 1
data2_dict = dict_generator(data2, task='IGT_SGT')

# ====================================================================
# Define the model
# ====================================================================
dual = DualProcessModel(num_trials=100, task='IGT_SGT', default_EV=0.0)
delta = ComputationalModels('delta', task='IGT_SGT', condition='Both', num_trials=100)
decay = ComputationalModels('decay', task='IGT_SGT', condition='Both', num_trials=100)

if __name__ == '__main__':
    # # ====================================================================
    # # Fit the model
    # # ====================================================================
    # dual_1 = dual.fit(data1_dict, num_iterations=100, weight_Gau='softmax', weight_Dir='softmax',
    #                      arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', num_t=2)
    # delta_1 = delta.fit(data1_dict, num_iterations=100)
    # decay_1 = decay.fit(data1_dict, num_iterations=100)
    dual_minmax_1 = dual.fit(data1_dict, num_iterations=200, weight_Gau='softmax', weight_Dir='softmax',
                                arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency_MinMax', num_t=1)

    # dual_2 = dual.fit(data2_dict, num_iterations=100, weight_Gau='softmax', weight_Dir='softmax',
    #                      arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', num_t=2)
    # delta_2 = delta.fit(data2_dict, num_iterations=100)
    # decay_2 = decay.fit(data2_dict, num_iterations=100)
    dual_minmax_2 = dual.fit(data2_dict, num_iterations=200, weight_Gau='softmax', weight_Dir='softmax',
                             arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency_MinMax', num_t=1)

    # # Save the results
    # dual_1.to_csv('./Data/OAYA/OAYA_pilot/dual.csv', index=False)
    # delta_1.to_csv('./Data/OAYA/OAYA_pilot/delta.csv', index=False)
    # decay_1.to_csv('./Data/OAYA/OAYA_pilot/decay.csv', index=False)
    dual_minmax_1.to_csv('./Data/OAYA/OAYA_pilot/dual_minmax.csv', index=False)

    # dual_2.to_csv('./Data/OAYA/OAYA_2022/dual.csv', index=False)
    # delta_2.to_csv('./Data/OAYA/OAYA_2022/delta.csv', index=False)
    # decay_2.to_csv('./Data/OAYA/OAYA_2022/decay.csv', index=False)
    dual_minmax_2.to_csv('./Data/OAYA/OAYA_2022/dual_minmax.csv', index=False)

    # ====================================================================
    # Read the results
    # ====================================================================
    dual_1 = pd.read_csv('./Data/OAYA/OAYA_pilot/dual.csv')
    delta_1 = pd.read_csv('./Data/OAYA/OAYA_pilot/delta.csv')
    decay_1 = pd.read_csv('./Data/OAYA/OAYA_pilot/decay.csv')
    dual_minmax_1 = pd.read_csv('./Data/OAYA/OAYA_pilot/dual_minmax.csv')

    dual_2 = pd.read_csv('./Data/OAYA/OAYA_2022/dual.csv')
    delta_2 = pd.read_csv('./Data/OAYA/OAYA_2022/delta.csv')
    decay_2 = pd.read_csv('./Data/OAYA/OAYA_2022/decay.csv')
    dual_minmax_2 = pd.read_csv('./Data/OAYA/OAYA_2022/dual_minmax.csv')

    print(f'Dual Process Model-1: {dual_1["BIC"].mean()}; Dual Process Model-2: {dual_2["BIC"].mean()}')
    print(f'Dual Process Model MinMax-1: {dual_minmax_1["BIC"].mean()}; Dual Process Model MinMax-2: {dual_minmax_2["BIC"].mean()}')
    print(f'Delta Model: {delta_1["BIC"].mean()}; Delta Model-2: {delta_2["BIC"].mean()}')
    print(f'Decay Model: {decay_1["BIC"].mean()}; Decay Model-2: {decay_2["BIC"].mean()}')

    # Extract the parameters
    dual_1 = parameter_extractor(dual_1, param_name=['t', 'alpha', 'subj_weight', 't2'])
    dual_minmax_1 = parameter_extractor(dual_minmax_1, param_name=['t', 'alpha', 'subj_weight'])
    delta_1 = parameter_extractor(delta_1, param_name=['t', 'alpha'])
    decay_1 = parameter_extractor(decay_1, param_name=['t', 'alpha'])

    dual_2 = parameter_extractor(dual_2, param_name=['t', 'alpha', 'subj_weight', 't2'])
    dual_minmax_2 = parameter_extractor(dual_minmax_2, param_name=['t', 'alpha', 'subj_weight'])
    delta_2 = parameter_extractor(delta_2, param_name=['t', 'alpha'])
    decay_2 = parameter_extractor(decay_2, param_name=['t', 'alpha'])

    # Convert string representation of list to actual list before exploding
    best_weight_1 = dual_1.copy()
    best_weight_1['best_weight'] = best_weight_1['best_weight'].apply(ast.literal_eval)
    best_weight_1['best_obj_weight'] = best_weight_1['best_obj_weight'].apply(ast.literal_eval)
    best_weight_1 = best_weight_1.explode(['best_weight', 'best_obj_weight'])


    # Rename the columns
    data1 = data1.rename(columns={'subjID': 'participant_id'})
    data2 = data2.rename(columns={'Subnum': 'participant_id'})

    # Merge the data
    data1_unique = data1.groupby('participant_id')['AgeGroup'].first().reset_index()
    dual_1 = pd.merge(dual_1, data1_unique, on='participant_id', how='left')
    dual_minmax_1 = pd.merge(dual_minmax_1, data1_unique, on='participant_id', how='left')

    data2_unique = data2.groupby('participant_id')['AgeGroup'].first().reset_index()
    dual_2 = pd.merge(dual_2, data2_unique, on='participant_id', how='left')
    dual_minmax_2 = pd.merge(dual_minmax_2, data2_unique, on='participant_id', how='left')

    # Combine the data
    dual_all = pd.concat([dual_1, dual_2], ignore_index=True)
    dual_all['participant_id'] = dual_all.index + 1
    dual_minmax_all = pd.concat([dual_minmax_1, dual_minmax_2], ignore_index=True)
    dual_minmax_all['participant_id'] = dual_minmax_all.index + 1

    # =====================================================================
    # T-test
    # =====================================================================
    # Perform an independent t-test
    df = dual_minmax_all
    var = 't'
    t_test = pg.ttest(df[df['AgeGroup'] == 'OA'][var],
                      df[df['AgeGroup'] == 'YA'][var], paired=False, alternative='two-sided', correction=True)
    print(f'{var}: {df[df['AgeGroup'] == 'OA'][var].mean()}')
    print(f'{var}: {df[df['AgeGroup'] == 'YA'][var].mean()}')
    print(t_test)

    # =====================================================
    # Plot the data
    # =====================================================
    # Plot the data
    plt.figure(figsize=(10, 5))

    # Create two subplots
    plt.subplot(1, 2, 1)
    sns.histplot(data=df[df['AgeGroup'] == 'OA'], x='subj_weight', bins=15)
    plt.title('Older Adults')
    plt.xlabel('Subject Weight')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.histplot(data=df[df['AgeGroup'] == 'YA'], x='subj_weight', bins=15)
    plt.title('Younger Adults')
    plt.xlabel('Subject Weight')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig('./Figures/OA_YA_subj_weight.png', dpi=600)
    plt.show()
