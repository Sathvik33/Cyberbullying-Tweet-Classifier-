import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Model")  # Model folder path
CSV_PATH = os.path.join(DATA_DIR, "cyberbullying_tweets.csv")

# Create Model directory if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", '', text)
    return text.strip()

df['tweet_text'] = df['tweet_text'].apply(clean_text)

X = df['tweet_text']
y = df['cyberbullying_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC())
])

param_grid = {
    'tfidf__max_features': [3000, 5000, 8000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'svm__C': [0.5, 1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

y_pred = grid.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the best model
model_path = os.path.join(MODEL_DIR, "best_cyberbullying_model.pkl")
joblib.dump(grid.best_estimator_, model_path)
print(f"Model saved to: {model_path}")