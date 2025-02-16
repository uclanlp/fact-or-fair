import os
import logging
import pandas as pd
import fairface

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the folders for each model
base_paths = {
    "flux-1.1-pro": "./sample_folder/flux-1.1-pro/",
    "dalle3": "./sample_folder/dalle3/",
    "midjourney": "./sample_folder/midjourney/",
    "sdxl-turbo": "./sample_folder/sdxl-turbo/"
}

# Create a list to store individual results
individual_results = []

# Process each model
for model_name, model_path in base_paths.items():
    logging.info(f"Starting analysis for model: {model_name}")

    # Analyze each image in the model folder
    for filename in sorted(os.listdir(model_path), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith(".png"):
            image_path = os.path.join(model_path, filename)
            try:
                logging.info(f"Processing image: {image_path}")
                results = fairface.predict(image_path)

                genders = []
                races = []

                for result in results:
                    race = result["predicted_race"]
                    gender = result["predicted_gender"]

                    genders.append(gender)
                    races.append(race)

                # Store individual results as dictionaries
                individual_results.append({
                    "Model": model_name,
                    "Image": filename,
                    "Genders": genders,
                    "Races": races
                })

            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")

    logging.info(f"Finished processing for model: {model_name}")

# Save individual results to CSV
individual_results_df = pd.DataFrame(individual_results)
individual_results_file = "./sample_fairface.csv"  # Save directly to this path
individual_results_df.to_csv(individual_results_file, index=False)

logging.info(f"Individual results saved to {individual_results_file}.")
