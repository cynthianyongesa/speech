#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.power import FTestAnovaPower

# Load the dataset from the specified file path
df = pd.read_excel('/Users/cynthianyongesa/Desktop/DATA/1_SPEECH/Demographics_Unique.xlsx')

# Map diagnosis groups according to user-defined categorization
diagnosis_mapping = {
    'Control': 'Neurotypical Group (NT)',       # Neurotypical control group
    'AD': 'Combined AD (cAD)',                  # Combined Alzheimer's Disease group
    'PossibleAD': 'Combined AD (cAD)',          # Possible AD classified under Combined AD
    'ProbableAD': 'Combined AD (cAD)',          # Probable AD classified under Combined AD
    'Vascular': 'Combined AD (cAD)',            # Vascular dementia classified under Combined AD
    'MCI': 'Mild Cognitive Impairment (MCI)',   # Mild Cognitive Impairment group
    'PPA': 'Primary Progressive Aphasia (PPA)', # Primary Progressive Aphasia group
    'Other': 'Other'                            # Other types of dementia or related conditions
}

# Apply the mapping to create a new 'Group' column
df['Group'] = df['Diagnosis'].map(diagnosis_mapping)

# Calculate mean and standard deviation for Age, Education, and Moca/MMSE by group
grouped_stats = df.groupby('Group').agg(
    Age_mean=('Age', 'mean'),                 # Mean age per group
    Age_sd=('Age', 'std'),                    # Standard deviation of age per group
    Education_mean=('Education', 'mean'),     # Mean years of education per group
    Education_sd=('Education', 'std'),        # Standard deviation of education years per group
    Moca_MMSE_mean=('Moca/MMSE', 'mean'),     # Mean Moca/MMSE score per group
    Moca_MMSE_sd=('Moca/MMSE', 'std')         # Standard deviation of Moca/MMSE scores per group
).reset_index()

# Calculate the number of females and males in each group
sex_distribution = df.groupby(['Group', 'Sex']).size().unstack(fill_value=0).reset_index()

# Combine the counts for 'Female' and 'female', and 'Male' and 'male'
# This handles potential discrepancies in case sensitivity (e.g., 'Female' vs 'female')
sex_distribution['Number_Female'] = sex_distribution.get('Female', 0) + sex_distribution.get('female', 0)
sex_distribution['Number_Male'] = sex_distribution.get('Male', 0) + sex_distribution.get('male', 0)

# Keep only the relevant columns for the number of females and males
sex_distribution = sex_distribution[['Group', 'Number_Female', 'Number_Male']]

# Merge the sex distribution data into the grouped statistics DataFrame
grouped_stats = pd.merge(grouped_stats, sex_distribution, on='Group')

# Save the final processed data to a new Excel file
output_file = 'Demographics_Unique_Processed.xlsx'
grouped_stats.to_excel(output_file, index=False)

# Print the final demographics summary to the console
print(grouped_stats)

# Inform the user that the processing is complete and the file is saved
print(f"Processing complete. Processed data saved to {output_file}.")

