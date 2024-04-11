import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK data downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Remove text within brackets
    text = re.sub(r'\[.*?\]', '', text)  # Removes anything inside square brackets
    text = re.sub(r'\(.*?\)', '', text)  # Removes anything inside round brackets
    text = re.sub(r'\{.*?\}', '', text)  # Removes anything inside curly brackets
    text = re.sub(r'\<.*?\>', '', text)  # Removes anything inside angle brackets

    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin words into a cleaned string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def clean_dataset(input_directory, output_directory, text_column):
    # Check and create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each CSV file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            print(f"Cleaning {file_path}...")

            # Load the dataset
            data = pd.read_csv(file_path)
            
            # Check if the specified text column exists in the dataset
            if text_column not in data.columns:
                raise ValueError(f"The specified column '{text_column}' does not exist in the dataset {filename}.")

            # Clean the text column
            data[text_column] = data[text_column].astype(str).apply(clean_text)

            # Remove rows where the text column is empty after cleaning
            data = data[data[text_column].str.strip().ne('')]
            
            # Save the cleaned dataset to a new CSV file in the output directory
            cleaned_file_path = os.path.join(output_directory, f"cleaned_{filename}")
            data.to_csv(cleaned_file_path, index=False)
            print(f"Cleaned dataset saved to {cleaned_file_path}")

# Paths to the input and output directories
input_directory = 'input'
output_directory = 'output'
text_column_name = 'text'  # Adjust to the name of the column containing the text to be cleaned

clean_dataset(input_directory, output_directory, text_column_name)
