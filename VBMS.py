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
    print("Posterior model probabilities per subject:\n", g_est.round(3))
    print("Expected model frequencies:", model_freq.round(3))
    print("Exceedance probabilities:", ex_probs_df.round(3))

    return alpha_est_df, model_freq, ex_probs_df

# ======================================================================================================================
# Gain-Loss Revision (Worthy & Hu, 2025)
# ======================================================================================================================
e1 = pd.read_csv('./Data/AllFitsABCDGLExp1_10_8_25.csv')
e2 = pd.read_csv('./Data/AllFitsABCDGLExp2_10_8_25.csv')

e1_alpha, e1_freq, e1_exceedance = vbms_quick_analysis(e1, metric='BIC')
e2_alpha, e2_freq, e2_exceedance = vbms_quick_analysis(e2, metric='BIC')

# # Save results
# e1_alpha.to_csv('./Exp1_VBMS_Alpha.csv')
# e1_freq.to_csv('./Exp1_VBMS_Frequency.csv')
# e1_exceedance.to_csv('./Exp1_VBMS_Exceedance.csv')
# e2_alpha.to_csv('./Exp2_VBMS_Alpha.csv')
# e2_freq.to_csv('./Exp2_VBMS_Frequency.csv')
# e2_exceedance.to_csv('./Exp2_VBMS_Exceedance.csv')
