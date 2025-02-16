import pandas as pd
import numpy as np

# Load the average KL Divergence and Entropy Ratio data
kl_df = pd.read_csv('./Test_Results/KL_Divergence/kl_divergence_average.csv')
entropy_df = pd.read_csv('./Test_Results/Entropy_Ratio/entropy_ratio_average.csv')

# Merge the two dataframes on Model
merged_df = pd.merge(kl_df, entropy_df, on='Model')

# Function to calculate fair score
def calculate_fair_score(entropy_ratio, kl_divergence):
    e_score = entropy_ratio
    kld_score = np.exp(-kl_divergence)
    
    fair_score = e_score + (1 - e_score) * kld_score
    return round(fair_score, 4), round(kld_score, 4)  # Return both fair_score and kld_score

# Calculate fair scores and KLD scores for gender and race
fair_scores = merged_df.apply(
    lambda row: calculate_fair_score(row['Gender Entropy Ratio'], row['Gender KL Divergence']),
    axis=1
)
merged_df['Gender Fair Score'], merged_df['Gender KLD Score'] = zip(*fair_scores)

fair_scores = merged_df.apply(
    lambda row: calculate_fair_score(row['Race Entropy Ratio'], row['Race KL Divergence']),
    axis=1
)
merged_df['Race Fair Score'], merged_df['Race KLD Score'] = zip(*fair_scores)

# Select relevant columns for fair score output
fair_score_df = merged_df[['Model', 'Gender Fair Score', 'Race Fair Score']]
fair_score_df.to_csv('./Test_Results/t2i_fair_score_subj.csv', index=False)

# Select relevant columns for KLD score output
kld_score_df = merged_df[['Model', 'Gender KLD Score', 'Race KLD Score']]
kld_score_df.to_csv('./Test_Results/t21_kld_score_subj.csv', index=False)

print("Fair scores saved to t2i_fair_score.csv.")
print("KLD scores saved to t21_kld_score_obj.csv.")
