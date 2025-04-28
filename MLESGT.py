import numpy as np
import pandas as pd
from scipy.optimize import minimize

data = pd.read_csv('C:/Users/bulic/OneDrive/CoronaData/JATOSWorkingData/CoronaWorkingData/Data/WSLL1.csv')
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
data['SGTBinChoice'] = data['SGTBinChoice'].map(mapping)

# Filter participants who have at least 4 unique choices
unique_counts = data.groupby('ID')['SGTBinChoice'].nunique()
valid_ids = unique_counts[unique_counts >= 4].index
data = data[data['ID'].isin(valid_ids)].reset_index(drop=True)


def compute_probability(E, c):
    """Compute choice probabilities based on expectations E and parameter c."""
    theta = 3 * c - 1
    Emax = max(E * theta)
    relative_weight = np.exp(theta * E - Emax)
    total_weight = sum(relative_weight)
    probs = relative_weight / total_weight
    return probs


def neg_log_likelihood(params, choices, outcomes):
    """Negative log-likelihood function to fit the model."""
    alpha, loss, phi, c = params
    E = np.zeros(4)
    log_likelihood = 0
    for i in range(len(choices)):
        choice = choices[i]
        outcome = outcomes[i]
        probs = compute_probability(E, c)
        log_likelihood += np.log(probs[choice - 1] + 1e-10)
        if outcome >= 0:
            utility = outcome ** alpha
        else:
            utility = -loss * abs(outcome) ** alpha
        delta = utility - E[choice - 1]
        E[choice - 1] += phi * delta
    return -log_likelihood


results = []

for pid, group in data.groupby('ID'):
    print('running participant', pid, '...')
    choices = group['SGTBinChoice'].values
    outcomes = group['SGTReward'].values
    best_res = None  # Track best optimization result

    for _ in range(150):  # Run minimize 150 times per participant
        # Generate random initial guess within reasonable bounds
        initial_params = [np.random.uniform(0.0, 1.0), np.random.uniform(0.1, 4.9),np.random.uniform(0.1, 4.9), np.random.uniform(0.0, 10.0)]

        # Optimize using MLE
        result = minimize(neg_log_likelihood, initial_params, args=(choices, outcomes),
                          method='L-BFGS-B', bounds=[(0.0, 1.0), (1.0e-5, 5.0), (1.0e-5, 1.0), (0.0, 10.0)])

        # Keep the best result (lowest neg_log_likelihood)
        if best_res is None or result.fun < best_res.fun:
            best_res = result

    # Store results
    logL_pvl = best_res.fun
    alpha_hat, gamma_hat, phi_hat, c_hat = best_res.x
    results.append(
        {'ID': pid, 'alpha': alpha_hat, 'gamma': gamma_hat, 'phi': phi_hat, 'c': c_hat, 'logL': logL_pvl})




results_df = pd.DataFrame(results)
results_df = results_df.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

#This commented out part is when I wast comparing the model fits. The Q-learning model was so bad that it wasn't of much use.
#
#
#from scipy.stats import chi2
#
#
# # Baseline Model: Q-learning for a task with 4 options
# def baseline_neg_log_likelihood(params, choices, rewards, n_actions=4):
#     alpha = params[0]  # alpha: learning rate
#
#     # Initialize Q-values for 4 options
#     Q = np.zeros(n_actions)  # 4 options (indexed 0, 1, 2, 3)
#     log_likelihood = 0
#
#     # Iterate over each trial
#     for t in range(len(choices)):
#         choice = choices[t]
#         reward = rewards[t]
#
#         # Update Q-values using the Q-learning update rule
#         Q[choice - 1] = (1 - alpha) * Q[choice - 1] + alpha * reward  # Update chosen option Q-value
#
#         # Softmax choice probability for 4 options
#         Q_max = np.max(Q)  # To prevent overflow in softmax
#         exp_Q = np.exp(Q - Q_max)  # Exponentiate the difference
#         prob = exp_Q[choice - 1] / np.sum(exp_Q)  # Probability of the chosen option
#
#         # Log-likelihood for the choice
#         log_likelihood += np.log(prob + 1e-10)  # Avoid log(0)
#
#     return -log_likelihood  # Return negative log-likelihood for minimization
#
#
# results_baseline = []
#
#
# for pid, group in data.groupby('ID'):
#     print('running participant', pid, '...')
#     choices = group['SGTBinChoice'].values
#     outcomes = group['SGTReward'].values
#
#     # Initial guesses for parameters
#     initial_params_baseline = [0.5]  # Initial guess for parameters (alpha, gamma)
#
#     # Optimize using MLE
#     result_baseline = minimize(baseline_neg_log_likelihood, initial_params_baseline, args=(choices, outcomes),
#                                method='L-BFGS-B', bounds=[(0, 1)])
#
#     # Store results
#     logL_baseline = -result_baseline.fun
#     alpha_hat = result_baseline.x
#     results_baseline.append({'ID': pid, 'alpha': alpha_hat, 'logL': logL_baseline})
#
# # Fit baseline model using MLE
#
# results_baseline_df = pd.DataFrame(results_baseline)
# results_baseline_df = results_baseline_df.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
# logL_baseline = results_baseline_df['logL'].sum()
# # Extract baseline model log-likelihood
#
sumlogL_pvl = results_df['logL'].sum()
#
# # Example PVL model log-likelihood from previous MLE fitting (already computed)
#
#
# # Number of parameters for each model
k_pvl = 4  # Example for PVL (alpha, gamma, phi, c)
# k_baseline = 1  # Example for baseline model (alpha)
#
# # Number of data points (trials per participant)
n = len(choices)
#
# # Compute AIC
AIC_pvl = 2 * k_pvl - 2 * sumlogL_pvl
# AIC_baseline = 2 * k_baseline - 2 * logL_baseline
#
# # Compute BIC
BIC_pvl = k_pvl * np.log(n) - 2 * sumlogL_pvl
# BIC_baseline = k_baseline * np.log(n) - 2 * logL_baseline
#
# # Likelihood Ratio Test (LRT)
# D = -2 * (logL_baseline - logL_pvl)  # Likelihood Ratio Statistic
# df = k_pvl - k_baseline  # Degrees of freedom (difference in parameters)
# p_value = 1 - chi2.cdf(D, df)  # p-value for the LRT
#
# # Results
print(f"PVL Model AIC: {AIC_pvl}, BIC: {BIC_pvl}")
# print(f"Baseline Model AIC: {AIC_baseline}, BIC: {BIC_baseline}")
# print(f"LRT Statistic: {D}, p-value: {p_value}")
# Optionally, save to a CSV file
#results_df.to_csv('PythonMLEResults.csv', index=False)


#PVL Model AIC: -58588.43135143288, BIC: -58578.01067068893 with 150 trials matching bounds
#PVL Model AIC: -59790.91224507286, BIC: -59783.0967345149 with 150 trials and gamma not in the equation