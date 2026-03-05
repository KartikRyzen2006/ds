#T-Test

import numpy as np
from scipy import stats

sample1 = np.array([10,12,11,13,12])  
# 🔴 CHANGE VALUES

sample2 = np.array([8,9,7,10,9])  
# 🔴 CHANGE VALUES

t_stat, p_value = stats.ttest_ind(sample1, sample2)

print("T-test p-value:", p_value)


# Chi-square Test

table = [[20,30],[10,40]]  
# 🔴 CHANGE TABLE VALUES

chi2, p, dof, expected = stats.chi2_contingency(table)

print("Chi-square p-value:", p)