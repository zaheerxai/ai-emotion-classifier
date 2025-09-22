## AI Emotion Classifier from Text

This project, developed as my Final Year Project (FYP) for my university in 2025, predicts emotions (joy, sadness, anger, fear, love, surprise) from text inputs using machine learning. Built with Python, scikit-learn, and NLTK for text preprocessing, it includes a Jupyter notebook for model training/evaluation and a Streamlit app for interactive emotion prediction. This work showcases my skills in AI and Python for the AI/Python Intern role at FOREO.

## Features

- Text Preprocessing: Cleaning, stopword removal, and lemmatization using NLTK.
- Models: Logistic Regression (0.89 accuracy) and Multinomial Naive Bayes (0.87 accuracy) for emotion classification.
- Interactive Demo: Streamlit app for real-time emotion predictions with confidence scores and visualizations.

## Project Structure

ai-emotion-classifier/

├── README.md

├── requirements.txt

├── PersonalitynEmotionAnalysisFromText.ipynb

├── app.py

├── emotion_model.pkl

├── tfidf_vectorizer.pkl

├── data/

│   └── combined_emotion.csv

└── .gitignore

## Setup

### Clone the Repository (or download ZIP):
git clone https://github.com/yourusername/ai-emotion-classifier.git

cd ai-emotion-classifier

### Create a Virtual Environment:
python -m venv venv

### Activate the Virtual Environment:
Windows: .\venv\Scripts\activate

macOS/Linux:source venv/bin/activate


### Install Dependencies:
pip install -r requirements.txt


### Download NLTK Data:
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"



## Usage

### Jupyter Notebook:
Open ai_emotion_classifier.ipynb in Jupyter or Colab.

Run cells to preprocess data/combined_emotion.csv, train models, and evaluate performance.


### Streamlit App:
Run the app:streamlit run app.py


Open http://localhost:8501 in your browser.

Enter text to predict emotions and view confidence scores with a bar chart.



## Example
Input: "I am so happy today!"
Output: Predicted Emotion: Joy 😊

### Probabilities:

**Joy**: 0.3552

**Anger**: 0.1621

**Love**: 0.1467

**Surprise**: 0.1367

**Sadness**: 0.1276

**Fear**: 0.0718

## Dataset
The combined_emotion.csv in the data/ folder contains text sentences labeled with emotions (joy, sadness, anger, fear, love, surprise). If not included in the repository, download it from [Google Drive link, if applicable, or contact me].

## Notes

- Developed as a university Final Year Project (2025) to demonstrate AI/ML proficiency.
- Dependencies pinned in requirements.txt for compatibility.
- Fixes applied: Preprocessing added to app.py predictions, notebook uses relative path (data/combined_emotion.csv), and "surprise" typo corrected.
- Model files (emotion_model.pkl, tfidf_vectorizer.pkl) included for immediate use with the Streamlit app.

## License

MIT License
