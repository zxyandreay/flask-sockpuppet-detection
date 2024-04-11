import pandas as pd
import os

def remove_duplicates_in_csv(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):  # Check for CSV files
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            # Load the CSV file
            df = pd.read_csv(input_file_path)

            # Remove duplicate rows
            df_cleaned = df.drop_duplicates()

            # Save the cleaned DataFrame to a new CSV file in the output folder
            df_cleaned.to_csv(output_file_path, index=False)
            print(f'Processed {filename} and saved to {output_file_path}')

# Specify the directories
input_folder = 'input'  # Folder where your input CSV files are located
output_folder = 'output'  # Folder where the cleaned CSV files will be saved

# Remove duplicate entries in the CSV files
remove_duplicates_in_csv(input_folder, output_folder)
