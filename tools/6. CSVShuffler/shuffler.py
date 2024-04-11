import pandas as pd
import os
import random

def randomize_csv_entries(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List and process each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"randomized_{filename}")

            # Load the dataset
            data = pd.read_csv(input_path)

            # Randomize the order of rows
            data = data.sample(frac=1).reset_index(drop=True)

            # Save the randomized dataset to a new CSV file
            data.to_csv(output_path, index=False)
            print(f"Randomized {filename} and saved to {output_path}")

# Paths to the input and output directories
input_folder = 'input'
output_folder = 'output'

# Randomize entries in all CSV files in the input folder
randomize_csv_entries(input_folder, output_folder)
