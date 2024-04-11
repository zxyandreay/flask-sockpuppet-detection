import pandas as pd
import os

def balance_csv_entries(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = []
    min_length = float('inf')

    # Load all CSV files and find the one with the least number of entries
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            min_length = min(min_length, len(df))
            csv_files.append((filename, df))

    # Balance each CSV file to have the same number of entries as the smallest one
    for filename, df in csv_files:
        if len(df) > min_length:
            df = df.sample(n=min_length)  # Randomly select entries to match the minimum length
        output_file_path = os.path.join(output_folder, filename)
        df.to_csv(output_file_path, index=False)
        print(f'Balanced {filename} and saved to {output_file_path}')

# Specify the directories
input_folder = 'input'  # Folder where your input CSV files are located
output_folder = 'output'  # Folder where the balanced CSV files will be saved

# Balance the entries in the CSV files
balance_csv_entries(input_folder, output_folder)
