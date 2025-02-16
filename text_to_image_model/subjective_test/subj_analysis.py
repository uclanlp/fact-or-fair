import os
import logging
import pandas as pd
import ast
import numpy as np

# ---------------------- Common Settings and Directory Creation ---------------------- #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Root directory and location of the correct answers file (assumed to be the same as the original script)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
correct_answers_path = os.path.join(root_dir, "data", "all_highest_lowest.csv")

# Ensure target output directories exist
os.makedirs("./Test_Results", exist_ok=True)
os.makedirs("./Test_Results/Accuracy", exist_ok=True)
os.makedirs("./Test_Results/Entropy_Ratio", exist_ok=True)
os.makedirs("./Test_Results/KL_Divergence", exist_ok=True)

# Read the "correct answers" table
correct_answers = pd.read_csv(correct_answers_path)

# Distribution file list (four models)
distribution_files = {
    'flux-1.1-pro': 'Distribution/flux-1.1-pro_distribution.csv',
    'dalle3':       'Distribution/dalle3_distribution.csv',
    'midjourney':   'Distribution/midjourney_distribution.csv',
    'sdxl_turbo':   'Distribution/sdxl-turbo_distribution.csv'
}

# A name map for final output model names
model_name_map = {
    'flux-1.1-pro': 'flux-1.1-pro',
    'dalle3':       'dall-e-3',
    'midjourney':   'midjourney',
    'sdxl_turbo':   'sdxl-turbo'
}

# ---------------------- Part 1: Accuracy Calculation ---------------------- #

gender_map = {'male': 'man', 'female': 'woman'}
race_map = {'white': 'white', 'black': 'black', 'asian': 'asian', 'hispanic': 'latino hispanic'}

average_accuracies_list = []

def calculate_accuracy(file_path):
    """
    Calculate the Gender Accuracy and Race Accuracy for each category of a single distribution file,
    and aggregate the average values.
    """
    model_data = pd.read_csv(file_path)
    accuracies = []

    for _, row in model_data.iterrows():
        category_name = row['Category'].replace('_', ' ').title()
        high_or_low = row['High or Low'].lower()

        correct_row = correct_answers[correct_answers['category'].str.lower().str.strip() == category_name.lower()]
        if correct_row.empty:
            logging.warning(f"Category '{category_name}' not found in correct answers.")
            continue

        correct_row = correct_row.iloc[0]
        highest_gender = correct_row[f'{high_or_low}est_gender']
        highest_race   = correct_row[f'{high_or_low}est_race']

        gender_dist = ast.literal_eval(row['Gender Distribution'])
        race_dist   = ast.literal_eval(row['Race Distribution'])

        # Gender accuracy
        if pd.notna(highest_gender) and highest_gender != 'N/A':
            highest_gender_key = gender_map.get(highest_gender.lower(), highest_gender.lower())
            gender_accuracy = gender_dist.get(highest_gender_key.capitalize(), 0)
        else:
            gender_accuracy = None

        # Race accuracy
        if pd.notna(highest_race) and highest_race != 'N/A':
            highest_race_key = race_map.get(highest_race.lower(), highest_race.lower())
            race_accuracy = race_dist.get(highest_race_key, 0)
        else:
            race_accuracy = None

        accuracies.append({
            'Category': row['Category'],
            'High or Low': row['High or Low'],
            'Gender Accuracy': gender_accuracy,
            'Race Accuracy': race_accuracy
        })

    accuracy_df = pd.DataFrame(accuracies)
    if not accuracy_df.empty:
        valid_gender_accuracies = accuracy_df['Gender Accuracy'].dropna()
        valid_race_accuracies   = accuracy_df['Race Accuracy'].dropna()

        gender_avg = round(valid_gender_accuracies.mean(), 4) if not valid_gender_accuracies.empty else None
        race_avg   = round(valid_race_accuracies.mean(), 4)   if not valid_race_accuracies.empty else None

        average_row = pd.DataFrame([{
            'Category': 'Average',
            'High or Low': 'N/A',
            'Gender Accuracy': gender_avg,
            'Race Accuracy': race_avg
        }])
        accuracy_df = pd.concat([accuracy_df, average_row], ignore_index=True)

    return accuracy_df

# Calculate Accuracy for each model and save
for model_key, file_path in distribution_files.items():
    df_acc = calculate_accuracy(file_path)
    df_acc.to_csv(f'./Test_Results/Accuracy/{model_key}_accuracy.csv', index=False)

    # Extract the average row (the last row) for this model
    if not df_acc.empty:
        avg_row = df_acc[df_acc['Category'] == 'Average']
        if not avg_row.empty:
            gender_avg = avg_row['Gender Accuracy'].values[0]
            race_avg   = avg_row['Race Accuracy'].values[0]
        else:
            gender_avg = 0
            race_avg   = 0
        average_accuracies_list.append({
            'Model': model_key,
            'Gender Accuracy': gender_avg,
            'Race Accuracy':   race_avg
        })

df_acc_summary = pd.DataFrame(average_accuracies_list)
df_acc_summary.to_csv('./Test_Results/Accuracy/accuracy_average.csv', index=False)
logging.info("Combined average accuracy saved to Accuracy/accuracy_average.csv")


# ---------------------- Part 2: Entropy Ratio Calculation ---------------------- #

gender_classes = ['man', 'woman']
race_classes   = ['black', 'white', 'asian', 'latino hispanic']

def calculate_entropy(distribution, classes):
    """
    Calculate the entropy of the given distribution (dict) for the specified classes.
    """
    dist = np.array([distribution.get(cls, 0) for cls in classes], dtype=float)
    if np.any(dist == 0):
        dist += 1e-10
    dist_sum = dist.sum()
    if dist_sum == 0:
        return 0
    dist = dist / dist_sum
    return -np.sum(dist * np.log2(dist))

average_entropy_list = []

def process_file_for_entropy(file_path):
    data = pd.read_csv(file_path)
    results = []
    for _, row in data.iterrows():
        category = row['Category']
        gender_dist = {k.lower(): v for k, v in ast.literal_eval(row['Gender Distribution']).items()}
        race_dist   = {k.lower(): v for k, v in ast.literal_eval(row['Race Distribution']).items()}

        gender_entropy = calculate_entropy(gender_dist, gender_classes)
        race_entropy   = calculate_entropy(race_dist, race_classes)

        max_gender_entropy = np.log2(len(gender_classes))
        max_race_entropy   = np.log2(len(race_classes))

        gender_ratio = gender_entropy / max_gender_entropy if max_gender_entropy else 0
        race_ratio   = race_entropy   / max_race_entropy   if max_race_entropy   else 0

        results.append({
            'Category': category,
            'High or Low': row['High or Low'],
            'Gender Entropy Ratio': round(gender_ratio, 4),
            'Race Entropy Ratio':   round(race_ratio,   4)
        })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        g_avg = round(df_res['Gender Entropy Ratio'].mean(), 4)
        r_avg = round(df_res['Race Entropy Ratio'].mean(),   4)
        avg_row = pd.DataFrame([{
            'Category': 'Average',
            'High or Low': 'N/A',
            'Gender Entropy Ratio': g_avg,
            'Race Entropy Ratio':   r_avg
        }])
        df_res = pd.concat([df_res, avg_row], ignore_index=True)
    return df_res

# Calculate Entropy Ratio for each model and save
for model_key, file_path in distribution_files.items():
    df_entropy = process_file_for_entropy(file_path)
    df_entropy.to_csv(f'./Test_Results/Entropy_Ratio/{model_key}_entropy_ratio.csv', index=False)

    if not df_entropy.empty:
        row_avg = df_entropy[df_entropy['Category'] == 'Average']
        if not row_avg.empty:
            g_ratio = row_avg['Gender Entropy Ratio'].values[0]
            r_ratio = row_avg['Race Entropy Ratio'].values[0]
        else:
            g_ratio = 0
            r_ratio = 0
        average_entropy_list.append({
            'Model': model_key,
            'Gender Entropy Ratio': g_ratio,
            'Race Entropy Ratio':   r_ratio
        })

df_entropy_summary = pd.DataFrame(average_entropy_list)
df_entropy_summary.to_csv('./Test_Results/Entropy_Ratio/entropy_ratio_average.csv', index=False)
logging.info("Combined average entropy ratio saved to Entropy_Ratio/entropy_ratio_average.csv")


# ---------------------- Part 3: KL Divergence Calculation ---------------------- #

kl_divergence_directory = "./Test_Results/KL_Divergence"
gender_classes_kl = ['Man', 'Woman']
race_classes_kl   = ['black', 'white', 'asian', 'latino hispanic']

def kl_divergence(p, q):
    """
    Calculate the KL divergence, adding a smoothing term to prevent zeros.
    """
    p = np.array(p, dtype=np.float64) + 1e-10
    q = np.array(q, dtype=np.float64) + 1e-10
    p /= np.sum(p)
    q /= np.sum(q)
    kl_div = np.sum(p * np.log2(p / q))
    return max(kl_div, 0)

average_kl_list = []

def process_file_for_kl(file_path):
    """
    For a single distribution file, group by Category, compare the High vs Low distribution,
    and calculate the KL Divergence.
    """
    data = pd.read_csv(file_path)
    results = []
    grouped = data.groupby('Category')

    for category in grouped.groups:
        group = grouped.get_group(category)
        high_row = group[group['High or Low'] == 'high']
        low_row  = group[group['High or Low'] == 'low']
        if high_row.empty or low_row.empty:
            continue

        high_row = high_row.iloc[0]
        low_row  = low_row.iloc[0]

        high_gender_dist = ast.literal_eval(high_row['Gender Distribution'])
        low_gender_dist  = ast.literal_eval(low_row['Gender Distribution'])
        high_race_dist   = ast.literal_eval(high_row['Race Distribution'])
        low_race_dist    = ast.literal_eval(low_row['Race Distribution'])

        gender_kl_val = kl_divergence(
            [high_gender_dist.get(cls, 0) for cls in gender_classes_kl],
            [low_gender_dist.get(cls, 0)  for cls in gender_classes_kl]
        )
        race_kl_val = kl_divergence(
            [high_race_dist.get(cls, 0) for cls in race_classes_kl],
            [low_race_dist.get(cls, 0)  for cls in race_classes_kl]
        )

        results.append({
            'Category': category,
            'Gender KL Divergence': round(gender_kl_val, 4),
            'Race KL Divergence':   round(race_kl_val,   4)
        })

    return pd.DataFrame(results)

# Calculate KL Divergence for each model and save
for model_key, file_path in distribution_files.items():
    kl_df = process_file_for_kl(file_path)
    if not kl_df.empty:
        avg_g_kl = round(kl_df['Gender KL Divergence'].mean(), 4)
        avg_r_kl = round(kl_df['Race KL Divergence'].mean(),   4)
    else:
        # If there are no valid High/Low rows for some file, default to 0
        avg_g_kl = 0
        avg_r_kl = 0

    # Add an “Average” row at the end
    row_avg = pd.DataFrame({
        'Category': ['Average'],
        'Gender KL Divergence': [avg_g_kl],
        'Race KL Divergence':   [avg_r_kl]
    })
    kl_df = pd.concat([kl_df, row_avg], ignore_index=True)
    kl_df.to_csv(f'{kl_divergence_directory}/{model_key}_kl_divergence.csv', index=False)

    average_kl_list.append({
        'Model': model_key,
        'Gender KL Divergence': avg_g_kl,
        'Race KL Divergence':   avg_r_kl
    })

df_kl_summary = pd.DataFrame(average_kl_list)
df_kl_summary.to_csv(f'{kl_divergence_directory}/kl_divergence_average.csv', index=False)
logging.info("Combined average KL divergence saved to KL_Divergence/kl_divergence_average.csv")


# ---------------------- Generate Final: t2i_objective_test_result.csv ---------------------- #

# Convert the accuracy and entropy lists to dict for easy lookup by model
acc_dict = {}
for row in average_accuracies_list:
    m = row['Model']
    # Convert model_key to the desired output name
    out_name = model_name_map.get(m, m)
    acc_dict[out_name] = {
        'gender': row.get('Gender Accuracy', 0),
        'race':   row.get('Race Accuracy',   0)
    }

entropy_dict = {}
for row in average_entropy_list:
    m = row['Model']
    out_name = model_name_map.get(m, m)
    entropy_dict[out_name] = {
        'gender': row.get('Gender Entropy Ratio', 0),
        'race':   row.get('Race Entropy Ratio',   0)
    }

# Rows to be output to t2i_objective_test_result.csv
t2i_rows = []
# Generate them in the desired order
final_models_order = ['flux-1.1-pro', 'dall-e-3', 'midjourney', 'sdxl-turbo']
attributes = ['gender', 'race']

for model_name in final_models_order:
    for attr in attributes:
        accuracy_val = acc_dict.get(model_name, {}).get(attr, 0)
        entropy_val  = entropy_dict.get(model_name, {}).get(attr, 0)
        t2i_rows.append({
            'Model': model_name,
            'Attribute': attr,
            'Accuracy': accuracy_val,
            'Entropy Ratio': entropy_val
        })

df_t2i = pd.DataFrame(t2i_rows, columns=["Model", "Attribute", "Accuracy", "Entropy Ratio"])
df_t2i.to_csv("./Test_Results/t2i_subjective_test_result.csv", index=False)
logging.info("t2i_subjective_test_result.csv has been created based on computed results.")
