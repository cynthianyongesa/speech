#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re

def create_output_folder(output_folder):
    """
    Creates an output folder if it doesn't already exist.

    Args:
    output_folder (str): Path to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def process_cha(cha_file_path, output_folder):
    """
    Processes a .cha file, extracts header information, and saves the relevant data to a .txt file.

    Args:
    cha_file_path (str): Path to the .cha file.
    output_folder (str): Path to the output folder where the .txt file will be saved.
    """
    # Read the content of the CHA file
    with open(cha_file_path, "r", encoding="utf-8") as cha_file:
        cha_content = cha_file.read()

    # Extract header information from the CHA content
    max_words = 0
    header_info = None

    for line in cha_content.split('\n'):
        if line.startswith('@ID:'):
            fields = line.split('|')
            total_words = len([word for word in fields if word.isalnum() or word.isalpha()])
            
            if total_words > max_words:
                max_words = total_words
                header_info = fields

    # Define the participant ID and create the output .txt file
    participant_id = os.path.splitext(os.path.basename(cha_file_path))[0]
    txt_file_path = os.path.join(output_folder, f"{participant_id}.txt")

    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        # Write header information
        txt_file.write(
            f"Participant's ID: {participant_id}\n"
            f"Age: {header_info[3]}\n"
            f"Sex: {header_info[4]}\n"
            f"Diagnosis: {header_info[5]}\n"
            f"MMSE: {header_info[8]}\n\n"
        )

        # Separate and clean lines from the CHA file
        speaker = ""
        all_lines = []

        for line in cha_content.split('\n'):
            if line.startswith('*INV:'):
                speaker = "INV"
                line = line[len('*INV:'):]
            elif line.startswith("*PAR:"):
                speaker = "PAR"
                line = line[len('*PAR:'):]
            elif line.startswith("@G:"):
                speaker = "TASK"
                line = line[len('@G:'):]
            else:
                continue

            # Clean the text in the line
            line = re.sub(r'[0-9#%*\[\]&<>_]', '', line)
            all_lines.append((speaker, line))

        # Write cleaned lines to the output file
        for speaker, line in all_lines:
            txt_file.write(f"{speaker}: {line}\n")

    print(f"Done processing {os.path.basename(cha_file_path)}, saved to {os.path.basename(txt_file_path)}")

def process_all_cha_files(input_folder, output_folder):
    """
    Processes all .cha files in a given input folder and saves the output in the specified output folder.

    Args:
    input_folder (str): Path to the input folder containing .cha files.
    output_folder (str): Path to the output folder where the processed .txt files will be saved.
    """
    # Get a list of all .cha files in the input folder
    cha_files = [file for file in os.listdir(input_folder) if file.endswith('.cha')]

    # Process each .cha file
    for cha_file in cha_files:
        cha_file_path = os.path.join(input_folder, cha_file)
        create_output_folder(output_folder)
        process_cha(cha_file_path, output_folder)
    
    print(f"FINISHED PROCESSING ALL CHA FILES IN {os.path.basename(input_folder)}")

# Define input and output folder paths
corpus = 'Pitt'
input_folder = os.path.join(os.getcwd(), f"Corpus/{corpus}")
output_folder = os.path.join(os.getcwd(), f"Corpus_TXT/{corpus}")

# Process all CHA files
process_all_cha_files(input_folder, output_folder)

