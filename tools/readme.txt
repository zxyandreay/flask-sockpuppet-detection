This folder contains a suite of tools designed for efficient and effective data preprocessing and management, specifically tailored for CSV and JSON data formats. Each tool is focused on a particular aspect of data preparation, ensuring a comprehensive approach to preparing your datasets for analysis or machine learning processes.

Contents of the Folder:

1. FORMAT:
   - This is the template structure used by all tools in the collection. It contains:
     - An 'input' folder: Place your raw data files here for processing.
     - An 'output' folder: After processing, the modified data files are saved here.
     - 'tool.py': A placeholder Python script file to be replaced with the actual code for each specific tool.
     - 'start.bat': A batch script to easily run 'tool.py', facilitating straightforward tool execution.

2. JSON2CSV:
   - Converts JSON files into CSV format, making it easier to work with in traditional data analysis and processing environments.

3. NLPCleaner:
   - Designed for natural language processing tasks, it cleans text data within CSV files by performing several preprocessing steps like removing special characters, normalizing case, and applying lemmatization.

4. CSVDeDup:
   - Removes duplicate records from CSV files to ensure data uniqueness, crucial for accurate analysis and machine learning model training.

5. CSVBalancer:
   - Balances the number of records across multiple CSV files, ensuring that each file has the same number of entries, typically aligning them with the smallest dataset in the collection.

6. CSVMerger:
   - Merges multiple CSV files into one, maintaining a similar header structure and adding a 'label' column to trace back to the original file source of each entry.

7. CSVShuffler:
   - Randomizes the order of entries in CSV files, which is vital for removing potential bias in sequential data and preparing datasets for unbiased analytical or machine learning tasks.

8. CSVReducer:
   - Reduces the number of entries in CSV files to a user-defined count by randomly removing excess records, allowing for the creation of manageable and specific-sized datasets.

Each tool in this folder is designed to operate independently, allowing users to address specific data preparation needs as required by their projects or analyses. The uniform structure provided by the 'FORMAT' template ensures consistency and ease of use across all tools in this collection.
