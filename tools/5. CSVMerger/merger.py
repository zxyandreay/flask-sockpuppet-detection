import pandas as pd
import os

def merge_csv_files_with_label(input_folder, output_folder, output_filename='merged.csv'):
    # Check and create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    merged_df = pd.DataFrame()  # Initialize an empty DataFrame for merging

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):  # Check for CSV files
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # Add a 'label' column indicating the source file
            df['label'] = os.path.splitext(filename)[0]

            # Merge with the overall DataFrame
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Save the merged DataFrame to a new CSV file in the output folder
    output_path = os.path.join(output_folder, output_filename)
    merged_df.to_csv(output_path, index=False)
    print(f'Merged CSV saved to {output_path}')

# Specify the directories
input_folder = 'input'  # Folder where your input CSV files are located
output_folder = 'output'  # Folder where the merged CSV file will be saved

# Merge the CSV files and add a label
merge_csv_files_with_label(input_folder, output_folder)
