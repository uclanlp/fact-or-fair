<div align= "center">
    <h1> ⚖️ Fact-or-Fair</h1>
</div>


<div align="center">
<img src="obj-cover.png" width="750px">
</div>
<div align="center">
<img src="subj-cover.png" width="750px">
</div>


## 📈 Leaderboard

| Large Language Model | Obj. $S_{fact}$ Gender | Obj. $S_{fact}$ Race | Obj. $S_{fact}$ Avg. | Subj. $S_{fair}$ Gender | Subj. $S_{fair}$ Race | Subj. $S_{fair}$ Avg. | Avg. Gender | Avg. Race | Avg. Avg. |
|----------------------|------------------------|----------------------|----------------------|-------------------------|-----------------------|-----------------------|-------------|-----------|-----------|
| 👑 **GPT-4o**        | 95.56                  | 54.62                | 75.09                | 98.39                   | 96.18                 | 97.29                 | 96.98       | 75.40     | 86.19     |
| LLaMA-3.2            | 96.67                  | 47.22                | 71.95                | 98.67                   | 97.20                 | 97.93                 | 97.67       | 72.21     | 84.94     |
| Qwen-2.5             | 91.11                  | 52.78                | 71.95                | 98.83                   | 96.40                 | 97.61                 | 94.97       | 74.59     | 84.79     |
| WizardLM-2           | 96.67                  | 44.44                | 70.56                | 99.17                   | 97.51                 | 98.34                 | 97.92       | 70.97     | 84.45     |
| Gemini-1.5           | 94.44                  | 44.44                | 69.44                | 98.13                   | 97.67                 | 97.90                 | 96.28       | 71.05     | 83.67     |
| GPT-3.5              | 84.44                  | 39.81                | 62.13                | 98.48                   | 96.28                 | 97.38                 | 91.46       | 68.04     | 79.75     |

| Text-to-Image Model | Obj. $S_{fact}$ Gender | Obj. $S_{fact}$ Race | Obj. $S_{fact}$ Avg. | Subj. $S_{fair}$ Gender | Subj. $S_{fair}$ Race | Subj. $S_{fair}$ Avg. | Avg. Gender | Avg. Race | Avg. Avg. |
|---------------------|------------------------|----------------------|----------------------|-------------------------|-----------------------|-----------------------|-------------|-----------|-----------|
| 👑 **DALL-E 3**     | 58.40                  | 30.33                | 44.37                | 96.35                   | 84.93                 | 90.64                 | 77.38       | 57.63     | 67.50     |
| Midjourney          | 48.90                  | 25.36                | 37.13                | 99.00                   | 75.99                 | 87.50                 | 73.95       | 50.68     | 62.31     |
| SDXL                | 51.97                  | 22.50                | 37.24                | 98.61                   | 74.40                 | 86.51                 | 75.29       | 48.45     | 61.87     |
| FLUX-1.1            | 49.07                  | 23.50                | 36.29                | 91.66                   | 30.36                 | 61.01                 | 70.37       | 26.93     | 48.65     |


## 📣 Updates
[Feb. 09, 2025] Published the arXiv preprint: [**arXiv:2502.05849**](https://arxiv.org/abs/2502.05849)


## ⚙️ Execution Process

### Dependencies and Installation
Please ensure your system meets the following requirements:
- **Python**: Version 3.10 or higher
- **Dependencies**: Install the required libraries by running the following command in the root directory:
  ```bash
  pip install -r requirements.txt
  ```

### Large Language Model

#### Configuring Environment Variables
The available models for testing are stored in **`large_language_model/models.py`**. Set the environment variables according to the model you choose to use:
```bash
export OPENAI_API_KEY="your_api_key_here"   # Leave empty if not used
export GEMINI_API_KEY="your_api_key_here"   # Leave empty if not used
export DEEPINFRA_TOKEN="your_api_key_here"  # Leave empty if not used
```

#### Testing
- **`large_language_model/objective_test/`**: Contains scripts for testing objective queries.
- **`large_language_model/subjective_test/`**: Contains scripts for testing subjective queries and daily scenarios.
- **`large_language_model/subjective_test/prompts_gen.py`**: A script for generating prompts related to daily scenarios.

1. Navigate to the project directory:
   ```bash
   cd large_language_model
   ```

2. Provide execution permissions to the test script:
   ```bash
   chmod +x run_llm_test.sh
   ```

3. Run the test script:
   ```bash
   ./run_llm_test.sh
   ```

Test results will be saved in the following directories:
- `objective_test/results/` for objective queries.
- `subjective_test/results/` for subjective queries.

#### Result Analysis
To visualize the test results, run:
```bash
python visualization.py
```
The visualizations will be saved in the `fig_results/` directory.

### Text-to-Image Model

#### Configuring Environment Variables
Set the environment variables according to the model you choose to use:
```bash
export OPENAI_API_KEY="your_api_key_here"   # Leave empty if not used
export MIDJOURNEY_API_SECRET="your_api_secret_here" # Leave empty if not used
export DEEPINFRA_TOKEN="your_api_key_here"  # Leave empty if not used
```

#### Testing
- **`text_to_image_model/objective_test/`**: Contains scripts for testing objective queries.
- **`text_to_image_model/subjective_test/`**: Contains scripts for testing subjective queries.

1. Navigate to the project directory:
   ```bash
   cd text_to_image_model
   ```

2. Provide execution permissions to the test script:
   ```bash
   chmod +x image_generate.sh
   chmod +x run_t2i_test.sh
   ```

3. Generate the images:
   ```bash
   ./image_generate.sh
   ```
   You need to manually place Midjourney images (due to a third-party setup), please do so after this step, in their correct folders.

4. Run the analysis:
   ```bash
   ./run_t2i_test.sh
   ```
   This script runs all analysis steps for both objective and subjective tests. Here, we use [FairFace](https://github.com/dchen236/FairFace) as the detection tool. The code comparing the accuracy of DeepFace and FairFace can be found in `detector_accuracy_test/`.


Test results will be saved in the following directories:
- `objective_test/Test_Results/` for objective queries.
- `subjective_test/Test_Results/` for subjective queries.

#### Result Analysis
To visualize the test results, run:
```bash
python visualization.py
```
The visualizations will be saved in the `Results_Visual/` directory.

## 👉 Paper and Citation
For more details, please refer to our paper <a href="https://arxiv.org/abs/2502.05849">here</a>.

If you find our paper&tool interesting and useful, please feel free to give us a star and cite us through:
```
@article{huang2025factorfairchecklistbehavioraltesting,
      title={Fact-or-Fair: A Checklist for Behavioral Testing of AI Models on Fairness-Related Queries}, 
      author={Jen-tse Huang and Yuhang Yan and Linqi Liu and Yixin Wan and Wenxuan Wang and Kai-Wei Chang and Michael R. Lyu},
      journal={arXiv: 2502.05849},
      year={2025}
}
```
