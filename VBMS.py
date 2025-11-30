import numpy as np
import pandas as pd
from utils.ComputationalModeling import vb_model_selection, compute_exceedance_prob

def vbms_quick_analysis(combined_BICs, metric='BIC'):
    bic_cols = [col for col in combined_BICs.columns if col.endswith(metric)]
    log_evidences = combined_BICs[bic_cols].values / (-2)
    k = log_evidences.shape[1]
    alpha0 = np.ones(k)  # uniform prior

    # Run VB model selection
    alpha_est, g_est = vb_model_selection(log_evidences, alpha0=alpha0, tol=1e-12, max_iter=50000)
    # Compute exceedance probabilities
    ex_probs = compute_exceedance_prob(alpha_est, n_samples=100000)

    # convert all to DataFrame for better readability
    alpha_est_df = pd.DataFrame(alpha_est, index=bic_cols).round(3)
    model_freq = pd.DataFrame((alpha_est / np.sum(alpha_est)).round(3), index=bic_cols, columns=['Frequency'])
    ex_probs_df = pd.DataFrame(ex_probs.round(3), index=bic_cols, columns=['Exceedance Probability'])

    print("Final alpha (Dirichlet parameters):", alpha_est_df.round(3))
    # print("Posterior model probabilities per subject:\n", g_est.round(3))
    print("Expected model frequencies:", model_freq.round(3))
    print("Exceedance probabilities:", ex_probs_df.round(3))

    return alpha_est_df, model_freq, ex_probs_df

# ======================================================================================================================
# Gain-Loss Revision (Worthy & Hu, 2025)
# ======================================================================================================================
e1 = pd.read_csv('./Data/AllFitsABCDGLExp1_10_8_25.csv')
e2 = pd.read_csv('./Data/AllFitsABCDGLExp2_10_8_25.csv')
e3 = pd.read_csv('./Data/AllFitsABCDGLExp3_10_20_25.csv')
model_order = ['deltaBIC', 'decayBIC', 'decayWinBIC', 'decayLossBIC', 'PVLDeltaBIC', 'PVLDecayBIC', 'PVPEDecayBIC', 'DeltaUncertaintyBIC']

e1_gains = e1[e1['Condition'] == 'Gains'].reset_index(drop=True)
e1_losses = e1[e1['Condition'] == 'Losses'].reset_index(drop=True)
e2_control = e2[e2['Condition'] == 'Control'].reset_index(drop=True)
e2_freq = e2[e2['Condition'] == 'Frequency'].reset_index(drop=True)
e3_control = e3[e3['Condition'] == 'LossesEF_Exp3'].reset_index(drop=True)
e3_freq = e3[e3['Condition'] == 'Losses_Exp3'].reset_index(drop=True)

e1_gains_alpha, e1_gains_freq, e1_gains_exceedance = vbms_quick_analysis(e1_gains, metric='BIC')
e1_losses_alpha, e1_losses_freq, e1_losses_exceedance = vbms_quick_analysis(e1_losses, metric='BIC')
e2_control_alpha, e2_control_freq, e2_control_exceedance = vbms_quick_analysis(e2_control, metric='BIC')
e2_freq_alpha, e2_freq_freq, e2_freq_exceedance = vbms_quick_analysis(e2_freq, metric='BIC')
e3_control_alpha, e3_control_freq, e3_control_exceedance = vbms_quick_analysis(e3_control, metric='BIC')
e3_freq_alpha, e3_freq_freq, e3_freq_exceedance = vbms_quick_analysis(e3_freq, metric='BIC')

# sort all results by model order
def sort_results(alpha_df, freq_df, exceedance_df, model_order):
    alpha_df = alpha_df.reindex(model_order).round(2)
    freq_df = freq_df.reindex(model_order).round(2)
    exceedance_df = exceedance_df.reindex(model_order).round(2)
    return alpha_df, freq_df, exceedance_df.round(2)

e1_gains_alpha, e1_gains_freq, e1_gains_exceedance = sort_results(e1_gains_alpha, e1_gains_freq, e1_gains_exceedance, model_order)
e1_losses_alpha, e1_losses_freq, e1_losses_exceedance = sort_results(e1_losses_alpha, e1_losses_freq, e1_losses_exceedance, model_order)
e2_control_alpha, e2_control_freq, e2_control_exceedance = sort_results(e2_control_alpha, e2_control_freq, e2_control_exceedance, model_order)
e2_freq_alpha, e2_freq_freq, e2_freq_exceedance = sort_results(e2_freq_alpha, e2_freq_freq, e2_freq_exceedance, model_order)
e3_control_alpha, e3_control_freq, e3_control_exceedance = sort_results(e3_control_alpha, e3_control_freq, e3_control_exceedance, model_order)
e3_freq_alpha, e3_freq_freq, e3_freq_exceedance = sort_results(e3_freq_alpha, e3_freq_freq, e3_freq_exceedance, model_order)

# # Save results
# e1_alpha.to_csv('./Exp1_VBMS_Alpha.csv')
# e1_freq.to_csv('./Exp1_VBMS_Frequency.csv')
# e1_exceedance.to_csv('./Exp1_VBMS_Exceedance.csv')
# e2_alpha.to_csv('./Exp2_VBMS_Alpha.csv')
# e2_freq.to_csv('./Exp2_VBMS_Frequency.csv')
# e2_exceedance.to_csv('./Exp2_VBMS_Exceedance.csv')


