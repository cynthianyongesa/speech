#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_excel('YOURFILE.xlsx')
df_all = data.copy()

# Convert diagnosis labels to lowercase for consistency
df_all['Diagnosis'] = df_all['Diagnosis'].str.lower()

# Step 1: Combine diagnoses into larger groups
ad_group = df_all[df_all['Diagnosis'].isin(['ad', 'probablead', 'vascular', 'possiblead'])]
mci_group = df_all[df_all['Diagnosis'] == 'mci']
control_group = df_all[df_all['Diagnosis'] == 'control']
other_group = df_all[df_all['Diagnosis'].isin(['other', 'Other'])]
ppa_group = df_all[df_all['Diagnosis'] == 'ppa']

# Step 2: Ensure unique participants by selecting one task per PID
ad_unique = ad_group.groupby('PID').apply(lambda x: x.sample(1)).reset_index(drop=True)
mci_unique = mci_group.groupby('PID').apply(lambda x: x.sample(1)).reset_index(drop=True)
control_unique = control_group.groupby('PID').apply(lambda x: x.sample(1)).reset_index(drop=True)
other_unique = other_group.groupby('PID').apply(lambda x: x.sample(1)).reset_index(drop=True)
ppa_unique = ppa_group.groupby('PID').apply(lambda x: x.sample(1)).reset_index(drop=True)

# Step 3: Combine all unique participant data into a single DataFrame
unique_df = pd.concat([ad_unique, mci_unique, control_unique, other_unique, ppa_unique], ignore_index=True)

# List of linguistic features to analyze
linguistic_features = [
    'Polarity', 'Subjectivity', 'Lexical_Diversity', 'verbs', 'nouns', 'pronouns',
    'adjectives', 'adverbs', 'interjections', 'determiners', 'conjunctions',
    'prepositions', 'auxiliary_verbs', 'particles', 'numbers', 'total_counts',
    'ocw', 'ccw', 'content_density', 'Personal_Deixis_Rate', 'Spatial_Deixis_Rate',
    'Temporal_Deixis_Rate', 'TTR', 'Brunets_Index', 'Honors_Statistic', 'Dale_Chall',
    'Flesch', 'Coleman_Liau_Index', 'Automated_Readability_Index', 'Reading_Time',
    'Syllables', 'Word_Count', 'Disfluencies', 'Syntactic_Complexity', 'RefRReal',
    'Mean_Sentence_Length', 'Std_Sentence_Length', 'Lexical_Density', 'Number_of_Sentences',
    'Function_Word_Count'
]

def perform_t_tests(group1, group2, features):
    """
    Perform t-tests between two groups for a list of linguistic features.

    Parameters:
    group1 (pd.DataFrame): DataFrame for the first group.
    group2 (pd.DataFrame): DataFrame for the second group.
    features (list): List of linguistic feature column names to test.

    Returns:
    list: List of tuples containing feature names and p-values for significant features.
    """
    significant_features = []

    for feature in features:
        group1_data = group1[feature].dropna()
        group2_data = group2[feature].dropna()
        
        # Perform t-test
        t_statistic, p_value = ttest_ind(group1_data, group2_data)
        
        # Check if p-value is significant (less than 0.05)
        if p_value < 0.05:
            significant_features.append((feature, p_value))
    
    return significant_features

# Define groups for pairwise comparisons
groups = {
    'AD': ad_unique,
    'MCI': mci_unique,
    'Control': control_unique,
    'Other': other_unique,
    'PPA': ppa_unique
}

# Perform t-tests for each pairwise comparison
comparisons = [
    ('AD', 'Control'),
    ('MCI', 'Control'),
    ('Other', 'Control'),
    ('PPA', 'Control'),
    ('AD', 'MCI'),
    ('AD', 'Other'),
    ('AD', 'PPA'),
    ('MCI', 'Other'),
    ('MCI', 'PPA'),
    ('Other', 'PPA')
]

# Store significant features for each comparison
significant_features_dict = {}

for group1_name, group2_name in comparisons:
    group1 = groups[group1_name]
    group2 = groups[group2_name]
    
    significant_features = perform_t_tests(group1, group2, linguistic_features)
    significant_features_dict[(group1_name, group2_name)] = significant_features

# Display significant features
for comparison, features in significant_features_dict.items():
    group1_name, group2_name = comparison
    print(f"Significant linguistic features for {group1_name} vs {group2_name}:")
    for feature, p_value in features:
        print(f"{feature}: p-value = {p_value}")
    print()

# Function to perform t-tests and return p-values
def perform_t_tests_with_p_values(group1, group2, features):
    """
    Perform t-tests between two groups and return a dictionary of p-values.

    Parameters:
    group1 (pd.DataFrame): DataFrame for the first group.
    group2 (pd.DataFrame): DataFrame for the second group.
    features (list): List of linguistic feature column names to test.

    Returns:
    dict: Dictionary of feature names and their corresponding p-values.
    """
    p_values = {}
    for feature in features:
        group1_data = group1[feature].dropna()
        group2_data = group2[feature].dropna()
        
        # Perform t-test
        t_statistic, p_value = ttest_ind(group1_data, group2_data)
        p_values[feature] = p_value
    
    return p_values

# Store p-values for each pairwise comparison
p_values_dict = {}

for group1_name, group2_name in comparisons:
    group1 = groups[group1_name]
    group2 = groups[group2_name]
    
    p_values = perform_t_tests_with_p_values(group1, group2, linguistic_features)
    p_values_dict[(group1_name, group2_name)] = p_values

# Convert p-values dictionary to DataFrame for export
p_values_df = pd.DataFrame(p_values_dict).T

# Export p-values to Excel for further analysis or inclusion in a manuscript
p_values_df.to_excel('p_values_comparison.xlsx')

