# Flask Sockpuppet Detection App

## Overview
This Flask application analyzes and predicts whether a given text input is a "sockpuppet" or not based on text features. A sockpuppet is an online identity used for purposes of deception within online communities. The app uses machine learning for text analysis, employing a RandomForestClassifier model trained on a pre-processed dataset.

## Getting Started

### Prerequisites
- Python 3.x
- Pip package manager
- Virtual environment (recommended)

### Installation
1. Clone the repository to your local machine:
```
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

## Contributing
Contributions to the project are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
Specify the license under which the project is released.
