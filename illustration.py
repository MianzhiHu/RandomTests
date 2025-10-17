import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import norm, linregress, t, ttest_ind, ttest_rel, f
from sympy.stats.rv import probability
from statsmodels.graphics.factorplots import interaction_plot


def grade_reporter(grades):
    max_grade = max(grades)
    min_grade = min(grades)
    range_grade = max_grade - min_grade
    mean_grade = np.mean(grades)
    median_grade = np.median(grades)
    mode_grade = grades[grades.count(grades)]
    std_grade = np.std(grades)
    print(f'max: {max_grade}, min: {min_grade}, range: {range_grade}, mean: {mean_grade}, median: {median_grade}, '
          f'mode: {mode_grade}, std: {std_grade}')

    # plot the data
    sns.set(style='white')
    sns.histplot(grades, kde=True, bins=10)
    plt.title('Grades')
    plt.xlabel('Grades')
    plt.ylabel('Frequency')
    plt.savefig('./Figures/grades_distribution.png', dpi=600)
    plt.show()

# # =============================================================================
# # draw a random probability
# # =============================================================================
# probability = [0.1, 0.3, 0.4, 0.2]
# events = ['A', 'B', 'C', 'D']
#
# sns.set(style='white')
# sns.barplot(x=events, y=probability, palette='viridis')
# plt.ylabel('Probability')
# plt.xlabel('Events')
# plt.title('Probability of Events')
# plt.show()
#
# # draw a standard normal distribution
# x = np.linspace(-4, 4, 1000)
# y = norm.pdf(x, loc=0, scale=1)  # mean=0, std deviation=1 for standard normal distribution
# sns.set(style='white')
# sns.lineplot(x=x, y=y)
# # draw vertical lines
# plt.axvline(x=0, color='r', linestyle='--')
# plt.axvline(x=1, color='g', linestyle='--')
# plt.axvline(x=-1, color='g', linestyle='--')
# plt.axvline(x=2, color='b', linestyle='--')
# plt.axvline(x=-2, color='b', linestyle='--')
# plt.axvline(x=3, color='y', linestyle='--')
# plt.axvline(x=-3, color='y', linestyle='--')
# plt.title('Standard Normal Distribution')
# plt.xlabel('X')
# plt.ylabel('Probability Density')
# plt.legend()
# sns.despine()
# plt.savefig('standard_normal_distribution.png')
# plt.show()
#
# # draw cdf of standard normal distribution
# y = norm.cdf(x, loc=0, scale=1)
# sns.set(style='white')
# sns.lineplot(x=x, y=y)
# plt.title('Standard Normal Distribution')
# plt.xlabel('X')
# plt.ylabel('CDF')
# plt.legend()
# plt.show()
#
# # draw a t-distribution
# x = np.linspace(-4, 4, 1000)
# y = t.pdf(x, df=23)
# sns.set(style='white')
# sns.lineplot(x=x, y=y)
# plt.title('T-Distribution at df=23')
# plt.xlabel('X')
# plt.ylabel('Probability Density')
# plt.axvline(x=0.716, color='r', linestyle='--')
# plt.axvline(x=-0.716, color='r', linestyle='--')
# plt.legend()
# sns.despine()
# plt.savefig('t_distribution.png')
# plt.show()
#
#
# # standardize the data
# data1 = [3, 4, 5, 4, 4, 4, 4]
# data2 = [3, 3, 3, 3, 3, 3, 4]
# t, p = ttest_ind(data1, data2)

# # =============================================================================
# # WA1 grade
# # =============================================================================
# wa1 = [37, 41.5, 37.5, 37, 40.5, 35.5, 43, 40, 35, 36.5, 36.5, 38, 39.5, 43.5, 41, 47, 38, 41, 38.5, 36, 39, 39.5, 42.5,
#        42]
#
# # calculate the drift rate
# slope, intercept, r_value, p_value, std_err = linregress(range(1, len(wa1) + 1), wa1)
#
# # multiply the numbers by 2
# wa1 = [x * 2 for x in wa1]
#
# max_wa1 = max(wa1)
# min_wa1 = min(wa1)
# range_wa1 = max(wa1) - min(wa1)
# mean_wa1 = np.mean(wa1)
# median_wa1 = np.median(wa1)
# mode_wa1 = wa1[wa1.count(wa1)]
# std_wa1 = np.std(wa1)
#
# # plot the data
# sns.set(style='white')
# sns.histplot(wa1, kde=True)
# plt.title('WA1 Grades')
# plt.xlabel('Grades')
# plt.ylabel('Frequency')
# plt.show()

# =============================================================================
# WA2 grade
# =============================================================================
# wa2 = [44, 45.5, 45, 40, 42, 44.5, 40.5, 46, 43, 37.5, 38.5, 34, 46, 38.5, 43.5, 37.5, 41, 43.5, 42.5, 32,
#        34, 31.5, 31.5, 43.5]
#
# # calculate the drift rate
# slope, intercept, r_value, p_value, std_err = linregress(range(1, len(wa2) + 1), wa2)
#
# # multiply the numbers by 2
# wa2 = [x * 2 for x in wa2]
#
# max_wa2 = max(wa2)
# min_wa2 = min(wa2)
# range_wa2 = max(wa2) - min(wa2)
# mean_wa2 = np.mean(wa2)
# median_wa2 = np.median(wa2)
# mode_wa2 = wa2[wa2.count(wa2)]
# std_wa2 = np.std(wa2)
# print(f'max: {max_wa2}, min: {min_wa2}, range: {range_wa2}, mean: {mean_wa2}, median: {median_wa2}, '
#       f'mode: {mode_wa2}, std: {std_wa2}')
#
# # plot the data
# sns.set(style='white')
# sns.histplot(wa2, kde=True)
# plt.title('WA2 Grades')
# plt.xlabel('Grades')
# plt.ylabel('Frequency')
# plt.show()
#
# # paired t-test
# print(ttest_ind(wa1, wa2))
# print(ttest_rel(wa1, wa2))
#
# # calculate t step by step
# mean_diff = np.mean(wa1) - np.mean(wa2)
# pooled_var = np.sqrt((np.var(wa1, ddof=1) / len(wa1)) + (np.var(wa2, ddof=1) / len(wa2)))
# t_stat = mean_diff / (pooled_var)
#
# # paired t
# mean_diff_paired = np.array(wa1) - np.array(wa2)
# std_paired = np.std(mean_diff_paired, ddof=1)
# t_stat_paired = mean_diff / (std_paired / np.sqrt(len(wa1)))

# ===============================================================================
# 2025 Fall WA1 grade
# ===============================================================================
wa1_2025_fall = [36, 35, 44, 38, 40, 32, 42, 30, 41, 41, 47, 39, 23, 34, 47]
wa1_2025_fall = [x * 2 for x in wa1_2025_fall]
grade_reporter(wa1_2025_fall)

# # =============================================================================
# # ANOVA illustration
# # =============================================================================
# # find the critical threshold
# alpha = 0.05
# dfn = 1
# dfd = 16
# threshold = f.ppf(1 - alpha, dfn, dfd)
# print(threshold)
#
# # find the p-value
# p_value = 1 - f.cdf(54.54, dfn, dfd)
# print(p_value)
#
# # create a F-distribution
# x = np.linspace(0, 10, 10000)
# y = f.pdf(x, dfn=2, dfd=3)
# y1 = f.pdf(x, dfn=2, dfd=117)
# sns.set(style='white')
# sns.lineplot(x=x, y=y, label='n=6', color='b')
# sns.lineplot(x=x, y=y1, label='n=120', color='orange')
# plt.title('F-Distribution')
# plt.xlabel('F')
# plt.ylabel('Probability Density')
# plt.axvline(x=9.5521, color='b', linestyle='--', label='p = .05 when n=6')
# plt.axvline(x=3.0738, color='orange', linestyle='--', label='p = .05 when n=120')
# plt.legend()
# sns.despine()
# plt.savefig('f_distribution.png')
# plt.show()
#
# # -----------------------------------------------------------------------------
# # draw illustrations for two-way ANOVA
# # -----------------------------------------------------------------------------
# # create a dataset
# data = {
#     'Exercise': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
#     'Sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
#     'Depression': [1, 0, 1, 2, 1, 0, 1, 2]
# }
# df = pd.DataFrame(data)
#
# # Plot the data with a line plot
# sns.set(style='white')
# sns.lineplot(x='Exercise', y='Depression', hue='Sex', marker='o', data=df, errorbar=None)
# plt.title('Interaction Effect of Exercise & Sex on Depression')
# plt.xlabel('Exercise')
# plt.ylabel('Depression')
# plt.legend(title='Sex')
# sns.despine()
# plt.savefig('two_way_anova.png')
# plt.show()

# # =============================================================================
# # WA3 grade
# # =============================================================================
# wa3 = [44, 46.5, 45.5, 43, 42, 25, 45.5, 37.5, 45.5, 41, 26, 49, 31, 41, 40.5, 41, 40, 41, 30.5, 43.5, 38, 48, 43]
# high_similarity = ['Isabella Barrera', 'Cody Carey', 'Josephine DeWalt', 'Tracy Garcia', 'Christopher Hernandez-Alfaro',
#                    'Nicolas Martinez', 'Suzette Santos', 'Emma Stephens']
#
# # calculate the drift rate
# slope, intercept, r_value, p_value, std_err = linregress(range(1, len(wa3) + 1), wa3)
#
# # multiply the numbers by 2
# wa3 = [x * 2 for x in wa3]
#
# max_wa3 = max(wa3)
# min_wa3 = min(wa3)
# range_wa3 = max(wa3) - min(wa3)
# mean_wa3 = np.mean(wa3)
# median_wa3 = np.median(wa3)
# mode_wa3 = wa3[wa3.count(wa3)]
# std_wa3 = np.std(wa3)

# # plot the data
# sns.set(style='white')
# sns.histplot(wa3, kde=True, bins=10)
# plt.title('WA3 Grades')
# plt.xlabel('Grades')
# plt.ylabel('Frequency')
# plt.show()


# # =============================================================================
# # Correlation
# # =============================================================================
# # create a low correlation dataset
# num_samples = 200
# x = np.linspace(0, 10, num_samples)
# y_low_positively_correlated = x + np.random.normal(0, 1, num_samples) + np.random.normal(0, 12, num_samples)
# y_low_negatively_correlated = -x + np.random.normal(0, 1, num_samples) + np.random.normal(0, 12, num_samples)
# y = np.random.normal(0, 1, num_samples)
# y_uncorrelated = y - np.dot(x, y) / np.dot(x, x) * x
#
#
# # create a high correlation dataset
# y_high_positively_correlated = x + np.random.normal(0, 1, num_samples)
# y_high_negatively_correlated = -x + np.random.normal(0, 1, num_samples)
#
# # calculate the correlation
# low_positively_correlated = np.corrcoef(x, y_low_positively_correlated)[0, 1]
# low_negatively_correlated = np.corrcoef(x, y_low_negatively_correlated)[0, 1]
# uncorrelated = np.corrcoef(x, y_uncorrelated)[0, 1]
# high_positively_correlated = np.corrcoef(x, y_high_positively_correlated)[0, 1]
# high_negatively_correlated = np.corrcoef(x, y_high_negatively_correlated)[0, 1]
#
# # plot the data
# sns.set(style='white')
# fig = plt.figure(figsize=(12, 6))
# gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
#
# # low positive correlation
# ax1 = plt.subplot(gs[0])
# sns.scatterplot(x=x, y=y_low_positively_correlated, ax=ax1)
# plt.title(f'Low Positive Correlation: {low_positively_correlated:.2f}')
#
# # low negative correlation
# ax2 = plt.subplot(gs[1])
# sns.scatterplot(x=x, y=y_low_negatively_correlated, ax=ax2)
# plt.title(f'Low Negative Correlation: {low_negatively_correlated:.2f}')
#
# # high positive correlation
# ax3 = plt.subplot(gs[2])
# sns.scatterplot(x=x, y=y_high_positively_correlated, ax=ax3)
# plt.title(f'High Positive Correlation: {high_positively_correlated:.2f}')
#
# # high negative correlation
# ax4 = plt.subplot(gs[3])
# sns.scatterplot(x=x, y=y_high_negatively_correlated, ax=ax4)
# plt.title(f'High Negative Correlation: {high_negatively_correlated:.2f}')
#
# # uncorrelated
# ax5 = plt.subplot(gs[4])
# sns.scatterplot(x=x, y=y_uncorrelated, ax=ax5)
# plt.title(f'Uncorrelated: {uncorrelated:.2f}')
#
# plt.tight_layout()
# plt.savefig('correlation.png')
# plt.show()
#
# # regression plot
# plt.clf()
# sns.set(style='white')
# fig = plt.figure(figsize=(12, 6))
# gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
#
# # low positive correlation
# ax1 = plt.subplot(gs[0])
# sns.regplot(x=x, y=y_low_positively_correlated, ax=ax1, line_kws={'color': 'red'})
# plt.title(f'Low Positive Correlation: {low_positively_correlated:.2f}')
#
# # low negative correlation
# ax2 = plt.subplot(gs[1])
# sns.regplot(x=x, y=y_low_negatively_correlated, ax=ax2, line_kws={'color': 'red'})
# plt.title(f'Low Negative Correlation: {low_negatively_correlated:.2f}')
#
# # high positive correlation
# ax3 = plt.subplot(gs[2])
# sns.regplot(x=x, y=y_high_positively_correlated, ax=ax3, line_kws={'color': 'red'})
# plt.title(f'High Positive Correlation: {high_positively_correlated:.2f}')
#
# # high negative correlation
# ax4 = plt.subplot(gs[3])
# sns.regplot(x=x, y=y_high_negatively_correlated, ax=ax4, line_kws={'color': 'red'})
# plt.title(f'High Negative Correlation: {high_negatively_correlated:.2f}')
#
# # uncorrelated
# ax5 = plt.subplot(gs[4])
# sns.regplot(x=x, y=y_uncorrelated, ax=ax5, line_kws={'color': 'red'})
# plt.title(f'Uncorrelated: {uncorrelated:.2f}')
#
# plt.tight_layout()
# plt.savefig('regression.png')
# plt.show()




