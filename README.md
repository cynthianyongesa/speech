# Linguistic Analysis for Early Dementia Detection

# Overview
This repository contains the code and scripts used in the study titled "Decoding the Language of Aging: Unveiling Linguistic Patterns Associated with Cognitive Impairment due to Alzheimer's Disease". The repository includes all the necessary scripts for data pre-processing, feature extraction, statistical analysis, and visualization. 

# Structure
- data/: Contains sample datasets (if any) or links to access the DementiaBank and AphasiaBank corpuses.
- scripts/: Includes all Python scripts used for data processing, analysis, and visualization.
- results/: Outputs of the analysis, including statistical summaries, figures, and tables.
- docs/: Documentation related to the project, including methodology details and references.

# Requirements
Python 3.8+
Required Python packages (specified in requirements.txt):
- pandas
- numpy
- statsmodels
- matplotlib
- seaborn
- scipy
- sklearn
- nltk
  
Install all dependencies using the following command:
- pip install -r requirements.txt

# Usage
## Data Preprocessing
The scripts/preprocessing.py script includes functions to clean and preprocess the audio transcripts and demographic data. Ensure that you have access to the DementiaBank and AphasiaBank datasets before running these scripts.

## Statistical Analysis
Use the scripts/statistical_analysis.py script to perform ANOVA, Tukey’s HSD tests, and Principal Component Analysis (PCA) on the linguistic features. Adjust parameters as necessary within the script.

## Visualization
Generate figures and visualizations by running scripts/visualization.py. The script produces charts for the key linguistic markers identified in the study.

## Results
The main results of the analysis, including significant linguistic markers and their correlations with cognitive assessments (MMSE, MoCA), are saved in the results/ directory. These results support the findings described in our published paper.

## Reproducibility
To reproduce the analysis:

1. Clone this repository.
2. Ensure all dependencies are installed (pip install -r requirements.txt).
3. Follow the steps outlined in the Usage section above.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Citation/Contact
If you use this code or data in your research, please cite our study. 
For questions or issues, please contact Cynthia Nyongesa at cnyongesa@ucsd.edu.


