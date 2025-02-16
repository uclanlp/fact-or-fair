import pandas as pd
import ast

# Load the CSV files
file1 = "sample_1.csv"
file2 = "sample_2.csv"
file3 = "sample_3.csv"

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Ensure the three DataFrames have the same structure and merge them
merged_df = pd.DataFrame({
    'Model': df1['Model'],
    'Image': df1['Image'],
    'Genders_1': df1['Genders'],
    'Genders_2': df2['Genders'],
    'Genders_3': df3['Genders'],
    'Races_1': df1['Races'],
    'Races_2': df2['Races'],
    'Races_3': df3['Races']
})

# Normalize list answers (e.g., "['male', 'female']" == "['female', 'male']")
def normalize_list(value):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return str(sorted(parsed))
        return value
    except (ValueError, SyntaxError):
        return value

# Perform majority voting for Genders and Races
def majority_vote(row, columns):
    answers = [normalize_list(row[col]) for col in columns]
    most_common = pd.Series(answers).mode()
    if len(most_common) == 1:
        return most_common[0]  # Return the majority answer
    else:
        return "N/A"  # Indicate no majority

merged_df['Final_Genders'] = merged_df.apply(lambda row: majority_vote(row, ['Genders_1', 'Genders_2', 'Genders_3']), axis=1)
merged_df['Final_Races'] = merged_df.apply(lambda row: majority_vote(row, ['Races_1', 'Races_2', 'Races_3']), axis=1)

# Count the number of N/A results for Genders and Races
genders_na_count = (merged_df['Final_Genders'] == "N/A").sum()
races_na_count = (merged_df['Final_Races'] == "N/A").sum()
print(f"Number of images with indeterminate Genders (N/A): {genders_na_count}")
print(f"Number of images with indeterminate Races (N/A): {races_na_count}")

# Save the final answers to a new CSV file
output_df = merged_df[['Model', 'Image', 'Final_Genders', 'Final_Races']]
output_df.rename(columns={'Final_Genders': 'Genders', 'Final_Races': 'Races'}, inplace=True)
output_df.to_csv("sample_answer.csv", index=False)
print("The final answers have been saved to 'sample_answer.csv'.")