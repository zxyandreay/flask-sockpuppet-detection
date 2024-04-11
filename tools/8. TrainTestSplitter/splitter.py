import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ensure input and output directories exist
input_dir = 'input'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # Load the dataset
        data_path = os.path.join(input_dir, filename)
        data = pd.read_csv(data_path)

        # Assume the class label is in the last column
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Split the data ensuring even distribution of classes and even numbers
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Make sure each split has an even number of examples
        # If not, remove excess from each
        if len(y_train) % 2 != 0:
            X_train = X_train[:-1]
            y_train = y_train[:-1]
        if len(y_test) % 2 != 0:
            X_test = X_test[:-1]
            y_test = y_test[:-1]

        # Combine features and labels back into a single DataFrame
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Save the split datasets
        train_data.to_csv(os.path.join(output_dir, f'train_{filename}'), index=False)
        test_data.to_csv(os.path.join(output_dir, f'test_{filename}'), index=False)

        print(f'Processed {filename}:')
        print(f'Training set saved to {output_dir}/train_{filename}')
        print(f'Testing set saved to {output_dir}/test_{filename}')
