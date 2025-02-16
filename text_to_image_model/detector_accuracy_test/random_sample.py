import os
import random
import shutil

def find_png(folder_path):
    """ Recursively find all PNG images in the given folder. """
    png_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files


def copy_random_pngs(models, objective_path, subjective_path, sample_folder='sample_folder', num_files=25):
    """ Collect PNG images from both objective and subjective folders for each model and copy a random sample. """
    
    # Ensure sample folder exists
    os.makedirs(sample_folder, exist_ok=True)
    
    for model in models:
        # Get all PNG files from both objective_test and subjective_test for the given model
        obj_folder = os.path.join(objective_path, "Images", model)
        subj_folder = os.path.join(subjective_path, "Images", model)

        obj_png_files = find_png(obj_folder) if os.path.exists(obj_folder) else []
        subj_png_files = find_png(subj_folder) if os.path.exists(subj_folder) else []

        all_png_files = obj_png_files + subj_png_files

        if not all_png_files:
            print(f"No PNG files found for model: {model}")
            continue

        # Select 25 random images
        selected_files = random.sample(all_png_files, min(num_files, len(all_png_files)))

        # Define destination folder
        dest_folder = os.path.join(sample_folder, model)
        os.makedirs(dest_folder, exist_ok=True)

        # Copy and rename selected images
        for i, file in enumerate(selected_files):
            new_filename = f"image_{i + 1}.png"  
            shutil.copy(file, os.path.join(dest_folder, new_filename))

        print(f"Finished copying {len(selected_files)} images to {dest_folder}.")


# Locate the current script's directory and move up one level to find objective_test and subjective_test
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Paths to objective_test and subjective_test
objective_test_path = os.path.join(parent_dir, "objective_test")
subjective_test_path = os.path.join(parent_dir, "subjective_test")

# Models to process
models = ["flux-1.1-pro", "dalle3", "midjourney", "sdxl-turbo"]

# Run the function
copy_random_pngs(models, objective_test_path, subjective_test_path)
