# Flask Sockpuppet Detection App

## Overview
This Flask application analyzes and predicts whether a given text input is a "sockpuppet" or not based on text features. A sockpuppet is an online identity used for purposes of deception within online communities. The app uses machine learning for text analysis, employing a RandomForestClassifier model trained on a pre-processed dataset.

## Getting Started

### Prerequisites
- Python 3.x
<<<<<<< HEAD
- Flask: A lightweight WSGI web application framework.
- Pandas: An open-source data analysis and manipulation tool.
- NumPy: A library for adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- NLTK: A leading platform for building Python programs to work with human language data (natural language processing).
- TextBlob: A library for processing textual data, providing simple APIs for common natural language processing (NLP) tasks.
- Scikit-learn: A machine learning library for the Python programming language.
=======
- Pip package manager
- Virtual environment (recommended)
>>>>>>> bae02cec (Create README.md)

### Installation
1. Clone the repository to your local machine:
```
<<<<<<< HEAD
git clone https://github.com/zxyandreay/flask-sockpuppet-detection.git
```
2. Run `install.bat` to create and activate a virtual environment, and install all the necessary prerequisites:
```
@echo off
cd /d %~dp0

python -m venv venv
call venv\Scripts\activate

pip install Flask pandas numpy textblob joblib gunicorn

pause
```


### Running the Application
1. Use `start.bat` to activate the virtual environment and run the application:
```
@echo off
cd /d %~dp0
call venv\Scripts\activate
python app.py  
=======
git clone https://your-repository-link-here.git
```
2. Navigate to the project directory:
```
cd flask-sockpuppet-detection
```
3. Create and activate a virtual environment:
- For Unix or MacOS:
  ```
  python3 -m venv env
  source env/bin/activate
  ```
- For Windows:
  ```
  py -m venv env
  .\env\Scripts\activate
  ```
4. Install the required dependencies:
```
pip install -r requirements.txt
```

### Running the Application
1. Start the Flask server:
```
python app.py
>>>>>>> bae02cec (Create README.md)
```
2. Open a web browser and navigate to `http://127.0.0.1:5000/`.

## Features
- Text preprocessing including tokenization, stop words removal, and lemmatization.
- Text analysis using TF-IDF vectorization.
- Sentiment analysis to extract polarity and subjectivity features.
- Sockpuppet prediction using RandomForestClassifier.
- Web interface for easy interaction with the model.

## Technology Stack
- Flask: Web framework for serving the application.
- Pandas and NumPy: For data manipulation and numerical operations.
- NLTK: For natural language processing, including text preprocessing.
- TextBlob: For performing sentiment analysis.
- Scikit-learn: For machine learning model building and vectorization.

<<<<<<< HEAD
## License
Specify the license under which the project is released.
=======
## Contributing
Contributions to the project are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
Specify the license under which the project is released.
>>>>>>> bae02cec (Create README.md)
