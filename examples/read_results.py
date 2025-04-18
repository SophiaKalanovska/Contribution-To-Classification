import pandas as pd 
# Visual comparison (using Matplotlib or Seaborn)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from innvestigate.faithfulnessCheck.calculate_distance import calculate_cosine_similarity

df = pd.read_csv('/Users/sophia/Documents/REVEAL_SAM/results.csv')

# Statistical summary
# print(df.describe())

normal_paters_pattern = '^REVEAL_ILSVRC2012_val_[0-9]{8}$'
blur_paters_pattern = '^REVEAL_ILSVRC2012_val_[0-9]{8}_blur$'
noise_big_paters_pattern = '^REVEAL_ILSVRC2012_val_[0-9]{8}_gausian_big$'
noise_small_paters_pattern = '^REVEAL_ILSVRC2012_val_[0-9]{8}_gausian_small$'

# Use the 'filter' method with the regex pattern
normal = df.filter(regex=normal_paters_pattern)
blur = df.filter(regex=blur_paters_pattern)
noise_big = df.filter(regex=noise_big_paters_pattern)
noise_small = df.filter(regex=noise_small_paters_pattern)


REVEAL_similarity_scores= []
REVEAL_similarity_scores_blur = []
REVEAL_similarity_scores_big =[]
REVEAL_similarity_scores_small = []

for col in normal.columns:
    explanation1 = normal[col].to_numpy()
    explanation2 = blur[col+'_blur'].to_numpy()
    explanation3 = noise_big[col+'_gausian_big'].to_numpy()
    explanation4 = noise_small[col+'_gausian_small'].to_numpy()
    
   
    mask = ~np.isnan(explanation1) 
    explanation1_flat_clean = explanation1[mask]
    mask = ~np.isnan(explanation2) 
    explanation2_flat_clean = explanation2[mask]
    mask = ~np.isnan(explanation3) 
    explanation3_flat_clean = explanation3[mask]
    mask = ~np.isnan(explanation4) 
    explanation4_flat_clean = explanation4[mask]

    # Calculate similarity score
    REVEAL_similarity_scores.append(calculate_cosine_similarity(explanation1_flat_clean, explanation1_flat_clean))
    REVEAL_similarity_scores_blur.append(calculate_cosine_similarity(explanation1_flat_clean, explanation2_flat_clean))
    REVEAL_similarity_scores_big.append(calculate_cosine_similarity(explanation1_flat_clean, explanation3_flat_clean))
    REVEAL_similarity_scores_small.append(calculate_cosine_similarity(explanation1_flat_clean, explanation4_flat_clean))


similarity_df_same =pd.DataFrame(REVEAL_similarity_scores, columns=['REVEAL_Similarity'])
similarity_df_blur = pd.DataFrame(REVEAL_similarity_scores_blur, columns=['REVEAL_Similarity'])
similarity_df_big = pd.DataFrame(REVEAL_similarity_scores_big, columns=['REVEAL_Similarity'])
similarity_df_small =pd.DataFrame(REVEAL_similarity_scores_small, columns=['REVEAL_Similarity'])



print(similarity_df_same.describe())
print(similarity_df_blur.describe())
print(similarity_df_big.describe())
print(similarity_df_small.describe())


normal_paters_pattern = '^LRP_ILSVRC2012_val_[0-9]{8}$'
blur_paters_pattern = '^LRP_ILSVRC2012_val_[0-9]{8}_blur$'
noise_big_paters_pattern = '^LRP_ILSVRC2012_val_[0-9]{8}_gausian_big$'
noise_small_paters_pattern = '^LRP_ILSVRC2012_val_[0-9]{8}_gausian_small$'

# Use the 'filter' method with the regex pattern
normal = df.filter(regex=normal_paters_pattern)
blur = df.filter(regex=blur_paters_pattern)
noise_big = df.filter(regex=noise_big_paters_pattern)
noise_small = df.filter(regex=noise_small_paters_pattern)



LRP_similarity_scores= []
LRP_similarity_scores_blur = []
LRP_similarity_scores_big = []
LRP_similarity_scores_small = []


for col in normal.columns:
    explanation1 = normal[col].to_numpy()
    explanation2 = blur[col+'_blur'].to_numpy()
    explanation3 = noise_big[col+'_gausian_big'].to_numpy()
    explanation4 = noise_small[col+'_gausian_small'].to_numpy()
    
   
    mask = ~np.isnan(explanation1) 
    explanation1_flat_clean = explanation1[mask]
    mask = ~np.isnan(explanation2) 
    explanation2_flat_clean = explanation2[mask]
    mask = ~np.isnan(explanation3) 
    explanation3_flat_clean = explanation3[mask]
    mask = ~np.isnan(explanation4) 
    explanation4_flat_clean = explanation4[mask]

    # Calculate similarity score
    LRP_similarity_scores.append(calculate_cosine_similarity(explanation1_flat_clean, explanation1_flat_clean))
    LRP_similarity_scores_blur.append(calculate_cosine_similarity(explanation1_flat_clean, explanation2_flat_clean))
    LRP_similarity_scores_big.append(calculate_cosine_similarity(explanation1_flat_clean, explanation3_flat_clean))
    LRP_similarity_scores_small.append(calculate_cosine_similarity(explanation1_flat_clean, explanation4_flat_clean))



LRP_similarity_df_same = pd.DataFrame(LRP_similarity_scores, columns=['LRP_Similarity'])
LRP_similarity_df_blur =  pd.DataFrame(LRP_similarity_scores_blur, columns=['LRP_Similarity'])
LRP_similarity_df_big =  pd.DataFrame(LRP_similarity_scores_big, columns=['LRP_Similarity'])
LRP_similarity_df_small =  pd.DataFrame(LRP_similarity_scores_small, columns=['LRP_Similarity'])



print(LRP_similarity_df_same.describe())
print(LRP_similarity_df_blur.describe())
print(LRP_similarity_df_big.describe())
print(LRP_similarity_df_small.describe())


concatenated_df = pd.concat([similarity_df_same, LRP_similarity_df_same], axis=1)
concatenated_df_blur = pd.concat([similarity_df_blur, LRP_similarity_df_blur], axis=1)
concatenated_df_big = pd.concat([similarity_df_big, LRP_similarity_df_big], axis=1)
concatenated_df_small = pd.concat([similarity_df_small, LRP_similarity_df_small], axis=1)

# blur = df.iloc[:, lambda x: x.index % 4 == 1]
# noise_big = df.iloc[:, lambda x: x.index % 4 == 2]
# noise_small = df.iloc[:, lambda x: x.index % 4 == 3]

# print(selected_columns.describe())

# Filter for odd rows (indices 1, 3, 5, ...)

# print(blur.describe())


sns.boxplot(data=df)
plt.title('Comparison of Explanation Methods')
plt.ylabel('Distance Score')
plt.show()

