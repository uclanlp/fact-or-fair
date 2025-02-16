#!/bin/bash

# image_generate.sh
# This script runs only the image-generation steps for both objective_test and subjective_test.

# 1) OBJECTIVE TEST: Image Generation
cd objective_test || exit
cd Image_Generation || exit

# Run generation scripts
python dalle3_generate.py
python deepinfra_generate.py
python midjouney_generate.py

# Return to text_to_image_model directory
cd ../..
echo "Finished objective_test image generation."

# 2) SUBJECTIVE TEST: Image Generation
cd subjective_test || exit
cd Image_Generation || exit

# Run generation scripts
python dalle3_generate.py
python deepinfra_generate.py
python midjouney_generate.py

# Return to text_to_image_model directory
cd ../..
echo "Finished subjective_test image generation."

echo "All generation scripts have completed."
echo "Need to place Midjourney images manually, please do so now in the correct directory."
