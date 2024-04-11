import pandas as pd
import json
import os

def json_to_csv(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each JSON file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")

            # Load JSON data
            with open(input_path, 'r') as file:
                json_data = json.load(file)

            # Convert JSON to DataFrame
            df = pd.json_normalize(json_data)

            # Save DataFrame to CSV
            df.to_csv(output_path, index=False)
            print(f"Converted {filename} to CSV.")

# Specify the input and output directories
input_folder = 'input'
output_folder = 'output'

# Convert all JSON files in the input folder to CSV files in the output folder
json_to_csv(input_folder, output_folder)
