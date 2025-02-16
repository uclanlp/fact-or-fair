import os
import logging
import pandas as pd
import fairface

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the folders for each model
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "Images")

base_paths = {
    "flux-1.1-pro": os.path.join(images_dir, "flux-1.1-pro"),
    "dalle3": os.path.join(images_dir, "dalle3"),
    "midjourney": os.path.join(images_dir, "midjourney"),
    "sdxl-turbo": os.path.join(images_dir, "sdxl-turbo")
}


# Define the updated categories (subfolder names)
categories = [
    "COVID-19_Mortality_Rate", "Cancer_Incidence_Rate", "Crime_Rate", "Diabetes_Rate",
    "Educational_Attainment", "Employment_Rate", "HIV_Rate", "Insurance_Coverage_Rate",
    "Homelessness_Rate", "Homeownership_Rate", "Influenza_Hospitalization_Rate", "Life_Expectancy",
    "Mortality_Rate", "Obesity_Rate", "Poverty_Rate", "Unemployment_Rate",
    "Volunteer_Rate", "Voter_Turnout_Rate", "Weekly_Income"
]

# Create directory to store results
os.makedirs("./Distribution", exist_ok=True)

# Process each model
for model_name, model_path in base_paths.items():
    distribution_results_list = []
    count_results_list = []

    logging.info(f"Starting analysis for model: {model_name}")

    # Process each category and high/low folders
    for category in categories:
        for level in ["high", "low"]:
            folder_path = os.path.join(model_path, category, level)
            
            if not os.path.exists(folder_path):
                logging.warning(f"Folder {folder_path} is missing. Skipping category.")
                continue

            gender_count = {}
            race_count = {}

            # Analyze each image in the high/low category folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)
                    try:
                        logging.info(f"Processing image: {image_path}")
                        results = fairface.predict(image_path)

                        if results is not None:  # Check if results are not None
                            face_count = len(results)
                            logging.info(f"Detected {face_count} faces in {image_path}")

                            # Count results for each face detected
                            for result in results:
                                race = result.get("predicted_race")
                                gender = result.get("predicted_gender")

                                if gender:
                                    gender_count[gender] = gender_count.get(gender, 0) + 1
                                if race:
                                    race_count[race] = race_count.get(race, 0) + 1
                        else:
                            logging.warning(f"No faces found in {image_path}")

                    except Exception as e:
                        logging.error(f"Error processing {image_path}: {e}")

            # Sort and calculate distribution
            sorted_gender_count = dict(sorted(gender_count.items(), key=lambda item: item[1], reverse=True))
            sorted_race_count = dict(sorted(race_count.items(), key=lambda item: item[1], reverse=True))
            
            total_gender = sum(sorted_gender_count.values())
            total_race = sum(sorted_race_count.values())
            
            gender_distribution = {k: round(v / total_gender, 2) for k, v in sorted_gender_count.items()} if total_gender else {}
            race_distribution = {k: round(v / total_race, 2) for k, v in sorted_race_count.items()} if total_race else {}

            # Append distribution results
            distribution_results_list.append({
                "Category": category,
                "High or Low": level,
                "Gender Distribution": str(gender_distribution),
                "Race Distribution": str(race_distribution)
            })

            # Append count results
            count_results_list.append({
                "Category": category,
                "High or Low": level,
                "Gender Count": str(sorted_gender_count),
                "Race Count": str(sorted_race_count)
            })

            logging.info(f"Finished processing category: {category}, level: {level}")

    # Save distribution results to CSV
    distribution_df = pd.DataFrame(distribution_results_list)
    distribution_file = f"./Distribution/{model_name}_distribution.csv"
    distribution_df.to_csv(distribution_file, index=False)

    # Save count results to CSV
    count_df = pd.DataFrame(count_results_list)
    count_file = f"./Distribution/{model_name}_count.csv"
    count_df.to_csv(count_file, index=False)

    logging.info(f"Results saved for model {model_name} - Distribution and Count CSVs.")
