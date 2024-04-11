import pandas as pd
import os

def reduce_csv_entries(input_folder, output_folder, target_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            if len(df) > target_count:
                df_reduced = df.sample(n=target_count)
                print(f'Reduced {filename} to {target_count} entries.')
            elif len(df) == target_count:
                df_reduced = df
                print(f'{filename} already has {target_count} entries.')
            else:
                print(f'{filename} has less than {target_count} entries and cannot be reduced.')
                df_reduced = df  # Keep the DataFrame unchanged

            output_file_path = os.path.join(output_folder, filename)
            df_reduced.to_csv(output_file_path, index=False)

# Get the target count of entries from the user
try:
    target_entry_count = int(input("Enter the desired number of entries per CSV file: "))
    assert target_entry_count > 0
except (ValueError, AssertionError):
    print("Invalid input. Please enter a positive integer.")
else:
    input_dir = 'input'  # Adjust to your input directory path
    output_dir = 'output'  # Adjust to your output directory path
    reduce_csv_entries(input_dir, output_dir, target_entry_count)
