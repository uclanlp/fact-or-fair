import pandas as pd
import ast

def load_csv(file_path):
    return pd.read_csv(file_path)

def calculate_error_rate(reference, detector, column):
    total_cases = 0
    error_cases = 0

    for _, ref_row in reference.iterrows():
        image = ref_row['Image']
        model = ref_row['Model']
        ref_list = ast.literal_eval(ref_row[column])
        # print(ref_list)

        # Find the corresponding row in the detector data with the same image and model
        det_row = detector[(detector['Image'] == image) & (detector['Model'] == model)]

        if det_row.empty:
            # If no corresponding row, all cases are errors
            error_cases += len(ref_list)
            total_cases += len(ref_list)
        else:
            det_list = ast.literal_eval(det_row.iloc[0][column])
            total_cases += len(ref_list)

            # Use set operations to calculate errors
            ref_set = set(ref_list)
            det_set = set(det_list)

            # Calculate mismatches and unmatched cases
            correct_cases = len(ref_set.intersection(det_set))

            if len(det_set) <= len(ref_set):
                error_cases += len(ref_set) - correct_cases
            else:
                error_cases += len(det_set) - correct_cases

    error_rate = error_cases / total_cases if total_cases > 0 else 0
    return error_rate

def main():
    # Load the CSV files
    sample_1 = load_csv('sample_answer.csv')
    sample_deepface = load_csv('sample_deepface.csv')
    sample_fairface = load_csv('sample_fairface.csv')

    # Calculate error rates
    gender_error_deepface = calculate_error_rate(sample_1, sample_deepface, 'Genders')
    race_error_deepface = calculate_error_rate(sample_1, sample_deepface, 'Races')

    gender_error_fairface = calculate_error_rate(sample_1, sample_fairface, 'Genders')
    race_error_fairface = calculate_error_rate(sample_1, sample_fairface, 'Races')

    # Print results
    print(f"DeepFace Gender Error Rate: {gender_error_deepface:.2%}")
    print(f"DeepFace Race Error Rate: {race_error_deepface:.2%}")
    print(f"FairFace Gender Error Rate: {gender_error_fairface:.2%}")
    print(f"FairFace Race Error Rate: {race_error_fairface:.2%}")

if __name__ == "__main__":
    main()