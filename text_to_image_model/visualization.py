import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------------
# Create / verify Results_Visual directory at the same level
# -------------------------------------------------------------------
output_dir = "Results_Visual"
os.makedirs(output_dir, exist_ok=True)


# ===================================================================
# PART 1: Scatter Plots of S_fact vs. S_fair (Objective and Subjective)
# ===================================================================

# ---------------------- COMMON SETTINGS FOR PART 1 ---------------------- #
model_colors = {
    'Flux-1.1-Pro': 'blue',
    'DALL-E 3': 'orange',
    'Midjourney': 'red',
    'SDXL-Turbo': 'green'
}
attribute_markers = {"gender": "^", "race": "o"}

model_name_replacements_df = {
    'flux-1.1-pro': 'Flux-1.1-Pro',
    'dall-e-3': 'DALL-E 3',
    'midjourney': 'Midjourney',
    'sdxl-turbo': 'SDXL-Turbo'
}
model_name_replacements_fair = {
    'flux-1.1-pro': 'Flux-1.1-Pro',
    'dalle3': 'DALL-E 3',
    'midjourney': 'Midjourney',
    'sdxl_turbo': 'SDXL-Turbo'
}

def plot_model_scores(df_result, df_fair, output_path, is_objective=True):
    """
    Given the test-result DataFrame and fair-score DataFrame, merge them,
    and produce a scatter plot for S_fact vs. S_fair.
    """
    # Replace model names for consistency
    df_result['Model'] = df_result['Model'].replace(model_name_replacements_df)
    df_fair['Model'] = df_fair['Model'].replace(model_name_replacements_fair)

    # Merge on 'Model'
    merged_df = pd.merge(df_result, df_fair, on='Model')

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each row of merged data
    for _, row in merged_df.iterrows():
        model = row['Model']
        attribute = row['Attribute']
        # S_fact = Accuracy * 100
        accuracy = 100 * row['Accuracy']
        # S_fair = Gender Fair Score or Race Fair Score, depending on 'Attribute'
        if attribute == "gender":
            s_fair = 100 * row['Gender Fair Score']
        else:  # attribute == "race"
            s_fair = 100 * row['Race Fair Score']

        ax.scatter(
            accuracy, s_fair,
            color=model_colors.get(model, 'gray'),
            marker=attribute_markers[attribute],
            s=500,
            edgecolor="black",
            alpha=0.5,
            label=model if attribute == 'race' else ""  # Label each model once (when attribute=='race')
        )

    # Add legend for markers indicating 'gender' or 'race'
    for attr_key, marker_symbol in attribute_markers.items():
        ax.scatter([], [], color='black', marker=marker_symbol, s=500,
                   label=attr_key.capitalize())

    # Axis labels
    ax.set_xlabel(r'$S_{\mathrm{fact}}$', fontsize=44)
    ax.set_ylabel(r'$S_{\mathrm{fair}}$', fontsize=44)

    # Ticks and limits
    ax.tick_params(axis='both', which='major', labelsize=34)
    ax.set_xlim(15, 65)  # Adjust as desired
    ax.set_ylim(35, 105) # Adjust as desired

    # Grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend
    legend = ax.legend(loc='lower left', fontsize=20)
    legend.get_frame().set_alpha(0.4)

    # Thicken the boundary of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # Final layout and save
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.show()

# ---------------------- OBJECTIVE TEST PLOT ---------------------- #
df_objective = pd.read_csv('./objective_test/Test_Results/t2i_objective_test_result.csv')
df_fair_obj = pd.read_csv('./objective_test/Test_Results/t2i_fair_score_obj.csv')
objective_output_path = os.path.join(output_dir, "model_scores_objective_test.png")
plot_model_scores(
    df_result=df_objective,
    df_fair=df_fair_obj,
    output_path=objective_output_path,
    is_objective=True
)

# ---------------------- SUBJECTIVE TEST PLOT ---------------------- #
df_subjective = pd.read_csv('./subjective_test/Test_Results/t2i_subjective_test_result.csv')
df_fair_subj = pd.read_csv('./subjective_test/Test_Results/t2i_fair_score_subj.csv')
subjective_output_path = os.path.join(output_dir, "model_scores_subjective_test.png")
plot_model_scores(
    df_result=df_subjective,
    df_fair=df_fair_subj,
    output_path=subjective_output_path,
    is_objective=False
)


# ===================================================================
# PART 2: Bar Charts of Fair Scores (Objective and Subjective)
# ===================================================================

def plot_fair_score_bar_chart(csv_path, output_filename, title):
    """
    Given the path to a fair-score CSV file, produce a bar chart of
    Gender Fair Score vs. Race Fair Score by Model, and save it.
    """
    fair_score_df = pd.read_csv(csv_path)

    # Replace model names
    fair_score_df['Model'] = fair_score_df['Model'].replace({
        'dalle3': 'dall-e-3',
        'midjourney': 'midjourney',
        'sdxl_turbo': 'stabilityai/sdxl-turbo',
        'flux-1.1-pro': 'black-forest-labs/FLUX-1.1-pro'
    })

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Gender Fair Score and Race Fair Score as grouped bars
    fair_score_df.plot(
        x='Model',
        y=['Gender Fair Score', 'Race Fair Score'],
        kind='bar',
        color=['orange', 'steelblue'],
        ax=ax
    )

    # Set plot title and labels
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Fair Score")
    ax.set_ylim(0, 1.0)  # Y-axis range from 0 to 1.0
    ax.legend(title="Category")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=0)

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the figure
    full_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_path)
    plt.show()

# ---------------------- OBJECTIVE FAIR SCORE BAR CHART ---------------------- #
objective_csv_path = "./objective_test/Test_Results/t2i_fair_score_obj.csv"
plot_fair_score_bar_chart(
    csv_path=objective_csv_path,
    output_filename="fair_score_plot_objective.png",
    title="Objective Test: Fair Score by Model and Category"
)

# ---------------------- SUBJECTIVE FAIR SCORE BAR CHART ---------------------- #
subjective_csv_path = "./subjective_test/Test_Results/t2i_fair_score_subj.csv"
plot_fair_score_bar_chart(
    csv_path=subjective_csv_path,
    output_filename="fair_score_plot_subjective.png",
    title="Subjective Test: Fair Score by Model and Category"
)


# ===================================================================
# PART 3: Trade-Off Plots (Objective and Subjective)
# ===================================================================

def f(a, k):
    """
    Entropy function for the trade-off curve.
    :param a: array-like, accuracy values (0 to 1)
    :param k: int, number of classes (2 for gender, 4 for race)
    :return: array-like, the calculated entropy ratio
    """
    return - (1 / np.log(k)) * (a * np.log(a) + (1 - a) * np.log((1 - a) / (k - 1)))

model_colors_tradeoff = {
    'Flux-1.1-Pro': 'blue',
    'DALL-E 3': 'orange',
    'Midjourney': 'red',
    'SDXL-Turbo': 'green'
}
attribute_markers_tradeoff = {"gender": "^", "race": "o"}

model_name_map_tradeoff = {
    'flux-1.1-pro': 'Flux-1.1-Pro',
    'dall-e-3': 'DALL-E 3',
    'midjourney': 'Midjourney',
    'sdxl-turbo': 'SDXL-Turbo'
}

def plot_trade_off(csv_path, output_filename, title):
    """
    Load data from csv_path, draw the max-entropy vs. accuracy curves for k=2 and k=4,
    then plot each model's (accuracy, entropy ratio).
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Replace model names
    df['Model'] = df['Model'].replace(model_name_map_tradeoff)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot max-entropy trade-off curves for k=2 and k=4
    for k_val, color in zip([2, 4], ["blue", "red"]):
        if k_val == 2:
            a_values = np.linspace(0.46, 1.0, 1000)
        else:
            a_values = np.linspace(0.22, 1.0, 1000)
        f_values = f(a_values, k_val)
        ax.plot(100 * a_values, 100 * f_values,
                label=rf'max $S_E$ (k={k_val})', 
                color=color)

    # Scatter each row's (Accuracy, Entropy Ratio)
    for _, row in df.iterrows():
        model = row['Model']
        attribute = row['Attribute']
        entropy = 100 * row['Entropy Ratio']
        accuracy = 100 * row['Accuracy']

        ax.scatter(
            accuracy, entropy,
            color=model_colors_tradeoff.get(model, "gray"),
            marker=attribute_markers_tradeoff.get(attribute, "o"),
            s=500,
            edgecolor="black",
            alpha=0.5,
            label=model if attribute == 'race' else ""
        )

    # Legend for attribute markers
    for attr_key, marker_symbol in attribute_markers_tradeoff.items():
        ax.scatter([], [], color='black', marker=marker_symbol, s=500,
                   label=attr_key.capitalize())

    # Customize plot
    ax.set_xlabel(r'$S_{\mathrm{fact}}$', fontsize=44)
    ax.set_ylabel(r'$S_{\mathrm{E}}$', fontsize=44)
    ax.tick_params(axis='both', which='major', labelsize=34)
    ax.set_xlim(0, 105)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add the legend
    legend = ax.legend(loc='lower left', fontsize=20)
    legend.get_frame().set_alpha(0.4)
    
    # Thicken axis lines
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    # Optional: set a title
    # ax.set_title(title, fontsize=26)

    # Final layout and save
    plt.tight_layout()
    full_output_path = os.path.join(output_dir, output_filename)
    plt.savefig(full_output_path)
    plt.show()

# ---------------------- TRADE-OFF: OBJECTIVE ---------------------- #
objective_csv_trade_path = "./objective_test/Test_Results/t2i_objective_test_result.csv"
plot_trade_off(
    csv_path=objective_csv_trade_path,
    output_filename="trade_off_objective.png",
    title="Objective Test: Accuracy vs. Entropy Ratio"
)

# ---------------------- TRADE-OFF: SUBJECTIVE ---------------------- #
subjective_csv_trade_path = "./subjective_test/Test_Results/t2i_subjective_test_result.csv"
plot_trade_off(
    csv_path=subjective_csv_trade_path,
    output_filename="trade_off_subjective.png",
    title="Subjective Test: Accuracy vs. Entropy Ratio"
)

# End of the combined script
