#!/bin/bash

# run_t2i_test.sh
# This script runs the analysis and testing steps after images have been generated.

# 1) OBJECTIVE TEST: Analysis & Testing
cd objective_test || exit

python image_analysis_fairface.py
python obj_analysis.py
python score_calculate.py
python distance_calculate.py

# Return to text_to_image_model directory
cd ..
echo "Finished objective_test analysis."

# 2) SUBJECTIVE TEST: Analysis & Testing
cd subjective_test || exit

python image_analysis_fairface.py
python subj_analysis.py
python score_calculate.py
python distance_calculate.py

# Return to text_to_image_model directory
cd ..
echo "Finished subjective_test analysis."

echo "All tests have completed."
