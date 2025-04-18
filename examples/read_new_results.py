import pandas as pd 
# Visual comparison (using Matplotlib or Seaborn)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


from innvestigate.faithfulnessCheck.calculate_distance import calculate_cosine_similarity

other_methods = pd.read_csv('/Users/sophia/Documents/REVEAL_SAM/examples/results_other_methods_vgg16.csv')
LRP_REVEAL_methods = pd.read_csv('/Users/sophia/Documents/REVEAL_SAM/examples/results_vgg16_partly.csv')


Same_other_methods = other_methods[(other_methods['Same_Classification__blur'] == True) | (other_methods['Same_Classification__gausian_small'] == True) | (other_methods['Same_Classification__gausian_big'] == True)]
Not_same_other_methods = other_methods[(other_methods['Same_Classification__blur'] == False) | (other_methods['Same_Classification__gausian_small'] == False) | (other_methods['Same_Classification__gausian_big'] == False)]

Same_LRP_REVEAL_methods = LRP_REVEAL_methods[(LRP_REVEAL_methods['Same_Classification__blur'] == True) | (LRP_REVEAL_methods['Same_Classification__gausian_small'] == True) | (LRP_REVEAL_methods['Same_Classification__gausian_big'] == True)]
Not_same_LRP_REVEAL_methods = LRP_REVEAL_methods[(LRP_REVEAL_methods['Same_Classification__blur'] == False) | (LRP_REVEAL_methods['Same_Classification__gausian_small'] == False) | (LRP_REVEAL_methods['Same_Classification__gausian_big'] == False)]


# --------------------------------------------------------------------------------------------------
data_to_plot_1 = Same_other_methods[['input_t_gradient__blur_euclidean', 'integrated_gradients__blur_euclidean', 'deconvnet_blur_euclidean', 'GuidedBackprop_blur_euclidean']]
data_to_plot_2 = Same_LRP_REVEAL_methods[['LRP__blur_euclidean', 'REVEAL__blur_euclidean']]


result = pd.concat([data_to_plot_1, data_to_plot_2])

# Renaming columns for the plot
result = result.rename(columns={'REVEAL__blur_euclidean': 'CTC values', 'LRP__blur_euclidean': 'LRP', 'input_t_gradient__blur_euclidean': 'Input x Gradient', 'integrated_gradients__blur_euclidean': 'Integrated Gradients', 'deconvnet_blur_euclidean': 'Deconvnet', 'GuidedBackprop_blur_euclidean': 'GuidedBackprop'})

# Plotting
sns.boxplot(data=result)
# plt.ylim(0, 2)
plt.title('Impact of Minor Blurring on Method Explanation Variability')
plt.ylabel('Distance Score')
plt.savefig("small_blur.png")
plt.show()

# --------------------------------------------------------------------------------------------------
data_to_plot_1 = Same_other_methods[['input_t_gradient__gausian_small_euclidean', 'integrated_gradients__gausian_small_euclidean', 'deconvnet_gausian_small_euclidean', 'GuidedBackprop_gausian_small_euclidean']]
data_to_plot_2 = Same_LRP_REVEAL_methods[['LRP__gausian_small_euclidean', 'REVEAL__gausian_small_euclidean']]

result = pd.concat([data_to_plot_1, data_to_plot_2])


# Renaming columns for the plot
result = result.rename(columns={'REVEAL__gausian_small_euclidean': 'CTC values', 'LRP__gausian_small_euclidean': 'LRP', 'input_t_gradient__gausian_small_euclidean': 'Input x Gradient', 'integrated_gradients__gausian_small_euclidean': 'Integrated Gradients', 'deconvnet_gausian_small_euclidean': 'Deconvnet', 'GuidedBackprop_gausian_small_euclidean': 'GuidedBackprop'})

# # Plotting
sns.boxplot(data=result)
# plt.ylim(0, 2)
plt.title('Impact of Minor Gaussian Noise (mean=0, std=0.05) on Method Explanation Variability')
plt.ylabel('Distance Score')
plt.show()
plt.savefig("small_gaussian_noise.png")


# --------------------------------------------------------------------------------------------------


small_change_LRP =  Same_LRP_REVEAL_methods[['LRP__gausian_small_euclidean']].dropna()
small_change_REVEAL =  Same_LRP_REVEAL_methods[['REVEAL__gausian_small_euclidean']].dropna()

big_change_LRP =  Not_same_LRP_REVEAL_methods[['LRP__gausian_big_euclidean']].dropna()
big_change_REVEAL =  Not_same_LRP_REVEAL_methods[['REVEAL__gausian_big_euclidean']].dropna()

# --------------------------------------------------------------------------------------------------

result_LRP = pd.concat([small_change_LRP, big_change_LRP])

result_LRP = result_LRP.rename(columns={'LRP__gausian_small_euclidean': 'Insignificant change', 'LRP__gausian_big_euclidean': 'Big change'})


t_statistic, p_value = stats.ttest_ind(small_change_LRP,
                                       big_change_LRP,
                                       equal_var=False) 



t_stat_scalar = t_statistic.item() if isinstance(t_statistic, np.ndarray) else t_statistic
p_val_scalar = p_value.item() if isinstance(p_value, np.ndarray) else p_value

print("T-statistic:", t_stat_scalar)
print("P-value:", p_val_scalar)


sns.boxplot(data=result_LRP)
plt.title('Small vs. Large Noise Perturbations for LRP: T-test and P-value')
plt.ylabel('Distance Score')
plt.text(x=0.05, y=0.95, s=f"T-statistic: {t_stat_scalar:.2f}\nP-value: {p_val_scalar:.2e}", 
         transform=plt.gca().transAxes, 
         verticalalignment='top')
plt.show()

# --------------------------------------------------------------------------------------------------

result_REVEAL = pd.concat([small_change_REVEAL, big_change_REVEAL])

result_REVEAL = result_REVEAL.rename(columns={'REVEAL__gausian_small_euclidean': 'Insignificant change', 'REVEAL__gausian_big_euclidean': 'Big change'})


t_statistic, p_value = stats.ttest_ind(small_change_REVEAL,
                                       big_change_REVEAL,
                                       equal_var=False) 

t_stat_scalar = t_statistic.item() if isinstance(t_statistic, np.ndarray) else t_statistic
p_val_scalar = p_value.item() if isinstance(p_value, np.ndarray) else p_value

print("T-statistic:", t_stat_scalar)
print("P-value:", p_val_scalar)

sns.boxplot(data=result_REVEAL)
plt.title('Small vs. Large Noise Perturbations for CTC values: T-test and P-value')
plt.ylabel('Distance Score')
plt.text(x=0.05, y=0.95, s=f"T-statistic: {t_stat_scalar:.2f}\nP-value: {p_val_scalar:.2e}", 
         transform=plt.gca().transAxes, 
         verticalalignment='top')
plt.show()

# --------------------------------------------------------------------------------------------------



small_change_LRP =  Same_LRP_REVEAL_methods[['LRP__blur_euclidean']].dropna()
small_change_LRP = small_change_LRP.rename(columns={'LRP__blur_euclidean': 'Insignificant change'})
small_change_REVEAL =  Same_LRP_REVEAL_methods[['REVEAL__blur_euclidean']].dropna()
small_change_REVEAL = small_change_REVEAL.rename(columns={'REVEAL__blur_euclidean': 'Insignificant change'})



big_change_LRP =  Not_same_LRP_REVEAL_methods[['LRP__blur_euclidean']].dropna()
big_change_LRP = big_change_LRP.rename(columns={'LRP__blur_euclidean': 'Big change'})
big_change_REVEAL =  Not_same_LRP_REVEAL_methods[['REVEAL__blur_euclidean']].dropna()
big_change_REVEAL = big_change_REVEAL.rename(columns={'REVEAL__blur_euclidean': 'Big change'})

# --------------------------------------------------------------------------------------------------

result_LRP = pd.concat([small_change_LRP, big_change_LRP])

t_statistic, p_value = stats.ttest_ind(small_change_LRP,
                                       big_change_LRP,
                                       equal_var=False) 



t_stat_scalar = t_statistic.item() if isinstance(t_statistic, np.ndarray) else t_statistic
p_val_scalar = p_value.item() if isinstance(p_value, np.ndarray) else p_value

print("T-statistic:", t_stat_scalar)
print("P-value:", p_val_scalar)


sns.boxplot(data=result_LRP)
plt.title('Small vs. Large Noise Perturbations for LRP: T-test and P-value')
plt.ylabel('Distance Score')
plt.text(x=0.05, y=0.95, s=f"T-statistic: {t_stat_scalar:.2f}\nP-value: {p_val_scalar:.2e}", 
         transform=plt.gca().transAxes, 
         verticalalignment='top')
plt.show()

# --------------------------------------------------------------------------------------------------

result_REVEAL = pd.concat([small_change_REVEAL, big_change_REVEAL])

t_statistic, p_value = stats.ttest_ind(small_change_REVEAL,
                                       big_change_REVEAL,
                                       equal_var=False) 

t_stat_scalar = t_statistic.item() if isinstance(t_statistic, np.ndarray) else t_statistic
p_val_scalar = p_value.item() if isinstance(p_value, np.ndarray) else p_value

print("T-statistic:", t_stat_scalar)
print("P-value:", p_val_scalar)

sns.boxplot(data=result_REVEAL)
plt.title('Small vs. Large Noise Perturbations for CTC values: T-test and P-value')
plt.ylabel('Distance Score')
plt.text(x=0.05, y=0.95, s=f"T-statistic: {t_stat_scalar:.2f}\nP-value: {p_val_scalar:.2e}", 
         transform=plt.gca().transAxes, 
         verticalalignment='top')
plt.show()

# --------------------------------------------------------------------------------------------------






# data_to_plot_1 = Not_same_other_methods[['input_t_gradient__gausian_big_euclidean', 'integrated_gradients__gausian_big_euclidean']]
# data_to_plot_2 = Not_same_LRP_REVEAL_methods[['LRP__gausian_big_euclidean', 'REVEAL__gausian_big_euclidean']]

# result = pd.concat([data_to_plot_1, data_to_plot_2])

# # Renaming columns for the plot
# result = result.rename(columns={'REVEAL__gausian_big_euclidean': 'CTC values', 'LRP__gausian_big_euclidean': 'LRP', 'input_t_gradient__gausian_big_euclidean': 'Input x Gradient', 'integrated_gradients__gausian_big_euclidean': 'Integrated Gradients'})


# # Plotting
# sns.boxplot(data=result)
# # plt.ylim(0, 2)
# plt.title('Change of explanation in REVEAL and LRP when presented with a big change in input')
# plt.ylabel('Distance Score')
# plt.show()



# data_to_plot_1 = Not_same_other_methods[['input_t_gradient__blur_euclidean', 'integrated_gradients__blur_euclidean']]
# data_to_plot_2 = Not_same_LRP_REVEAL_methods[['LRP__blur_euclidean', 'REVEAL__blur_euclidean']]

# result = pd.concat([data_to_plot_1, data_to_plot_2])

# # Renaming columns for the plot
# result = result.rename(columns={'REVEAL__blur_euclidean': 'CTC values', 'LRP__blur_euclidean': 'LRP', 'input_t_gradient__blur_euclidean': 'Input x Gradient', 'integrated_gradients__blur_euclidean': 'Integrated Gradients'})


# # Plotting
# sns.boxplot(data=result)
# # plt.ylim(0, 2)
# plt.title('Change of explanation in REVEAL and LRP when presented with a big blur in input')
# plt.ylabel('Distance Score')
# plt.show()


# sns.boxplot(data=df)
# plt.title('Comparison of Explanation Methods')
# plt.ylabel('Distance Score')
# plt.show()
