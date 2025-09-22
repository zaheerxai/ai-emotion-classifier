# AI Emotion Classifier from Text
### An NLP project for emotion detection from text

This project uses machine learning to classify emotions from text. The model identifies key emotions such as joy, sadness, anger, fear, love, and surprise. Developed with Python, scikit-learn, and NLTK, the project includes a Jupyter notebook for model training and a Streamlit application for interactive, real-time emotion prediction.

## Features

- **Text Preprocessing**: Comprehensive text cleaning, stopword removal, and lemmatization using the NLTK library.
- **Machine Learning Models**: Includes trained models (Logistic Regression, Multinomial Naive Bayes) with accuracies of 89% and 87% respectively.
- **Interactive Application**: A Streamlit web app for predicting emotions from user input, complete with confidence scores and a data visualization bar chart.

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

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-emotion-classifier.git
cd ai-emotion-classifier
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
Launch a Python interpreter and run the following commands:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Using the Jupyter Notebook
Open the ai_emotion_classifier.ipynb notebook in Jupyter or a similar environment. Run the cells to explore the data, preprocess the text, train the models, and evaluate their performance.

### Using the Streamlit App
To launch the interactive web application, run the following command from your terminal:
```bash
streamlit run app.py
```
The application will open in your browser at http://localhost:8501, allowing you to enter text and get real-time emotion predictions.

## Example
The model predicts the most likely emotion along with confidence scores.

### Example 1: Joy
Input: "I am so happy today!"

Output: Predicted Emotion: **Joy 😊**

**Confidence Scores:**

**Joy**: 0.3552

**Anger**: 0.1621

**Love**: 0.1467

**Surprise**: 0.1367

**Sadness**: 0.1276

**Fear**: 0.0718

### Example 2: Sadness
Input: "I lost my phone and my keys."

Output: Predicted Emotion: **Sadness 😔**

**Confidence Scores:**

**Sadness**: 0.4512

**Anger**: 0.2205

**Fear**: 0.1801

**Surprise**: 0.0910

**Love**: 0.0387

**Joy:** 0.0185

## Dataset
The data/combined_emotion.csv file contains the dataset used for training, featuring text entries labeled with their corresponding emotions.

## License

This project is licensed under the MIT License.
