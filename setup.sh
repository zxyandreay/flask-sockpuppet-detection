#!/bin/bash
# Navigate to the script's directory
cd "$(dirname "$0")"

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required Python packages
pip install Flask pandas numpy textblob joblib scikit-learn

# Pause execution to view output, press any key to continue
read -p "Press any key to continue . . . " -n1 -s
