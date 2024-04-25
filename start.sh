#!/bin/bash
# Navigate to the directory where the script resides
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python app.py

# Pause the script execution, press any key to continue
read -p "Press any key to continue . . . " -n1 -s
echo
