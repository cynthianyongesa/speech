#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import glob

# Define input and output folders
corpus = 'Pitt'
input_folder = os.path.join(os.getcwd(), f"Corpus_TXT/{corpus}")
output_folder = os.path.join(os.getcwd(), f"corpus_TXT/{corpus}")

def process_text_file(file_path):
    """
    Process a single .txt file and extract relevant data.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        pd.DataFrame: DataFrame containing extracted data.
    """
    with open(file_path, 'r') as file:
        txt_data = file.read()
    
    lines = txt_data.split('\n')
    
    data = {'PID': [], 'Age': [], 'Sex': [], 'Diagnosis': [], 'MMSE': [], 'Task': [], 'Text': []}
    current_task = None
    current_text = []

    for line in lines:
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            
            if key == "Participant's ID":
                data['PID'].append(value)
            elif key == "Age": 
                data['Age'].append(value)
            elif key == "Sex":
                data['Sex'].append(value)
            elif key == "Diagnosis": 
                data['Diagnosis'].append(value)
            elif key == "MMSE":
                data['MMSE'].append(value)
            elif key == "TASK":
                # Append previous task's text before starting a new task
                if current_task is not None: 
                    data['Task'].append(current_task)
                    data['Text'].append(' '.join(current_text))
                
                # Start new task
                current_task = value
                current_text = []
            else: 
                current_text.append(line.strip())

    # Append the last task's text
    data['Task'].append(current_task)
    data['Text'].append(' '.join(current_text))
    
    # Ensure all columns are of equal length
    max_length = max(len(data[key]) for key in data)
    for key in data:
        data[key] += [''] * (max_length - len(data[key]))

    df = pd.DataFrame(data)
    return df

def process_folder(folder_path, output_file):
    """
    Process all .txt files in the specified folder and save the combined data to an Excel file.

    Args:
        folder_path (str): Path to the folder containing .txt files.
        output_file (str): Path to the output Excel file.

    Returns:
        pd.DataFrame: Combined DataFrame containing all processed data.
    """
    df_list = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            df = process_text_file(file_path)
            df_list.append(df)
    
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        # Clean column names by removing special characters
        final_df.columns = final_df.columns.str.replace('[^\w\s]', '', regex=True)
        # Replace any remaining special characters in the data
        final_df.replace('[^\w\s]', '_', regex=True, inplace=True)
        final_df.to_excel(output_file, index=False)
        return final_df
    else:
        print("No text files found in the specified folder.")
        return None

def combine_excel_files(pattern, output_file):
    """
    Combine multiple Excel files into a single Excel file.

    Args:
        pattern (str): File pattern to match Excel files (e.g., '*.xlsx').
        output_file (str): Path to the output combined Excel file.

    Returns:
        None
    """
    xl_files = glob.glob(pattern)
    dfs = [pd.read_excel(file) for file in xl_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_excel(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    print(combined)

# Process the folder containing .txt files and save the results to an Excel file
if __name__ == "__main__":
    corpus = 'Pitt'
    folder_path = os.path.join(os.getcwd(), f"Corpus_TXT/{corpus}")
    output_file = 'clean_pitt_files.xlsx'

    result_df = process_folder(folder_path, output_file)

    if result_df is not None:
        print(result_df)

    # Uncomment the following lines if you need to combine multiple Excel files
    # pattern = os.path.join(os.getcwd(), 'clean_delaware_*.xlsx')
    # output_combined_file = 'combined_delaware.xlsx'
    # combine_excel_files(pattern, output_combined_file)
