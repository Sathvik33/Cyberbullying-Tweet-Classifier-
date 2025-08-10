"""Cyberbullying Tweet Classifier"""
This project builds a machine learning model to detect and classify types of cyberbullying in tweets. The app lets users input tweet text and instantly get predictions about the type of cyberbullying present.

Features
Cleans and preprocesses tweet text

Uses TF-IDF vectorization and LinearSVC classifier

Hyperparameter tuning with GridSearchCV

Saves the best model for deployment

Interactive, user-friendly Streamlit web app for classification

Explains "Other Bullying" category for better user understanding

Usage
Training the Model
(If you want to retrain the model yourself)

bash
Copy code
python train_model.py
Running the Streamlit App
bash
Copy code
streamlit run app.py
This will open the web app in your browser where you can enter tweet text and get predictions.

File Structure
train_model.py — Script for training the model and saving it

app.py — Streamlit app code for the GUI

Model/best_cyberbullying_model.pkl — Saved trained model

Data/cyberbullying_tweets.csv — Dataset CSV file

requirements.txt — Required Python packages

Model Details
Model: LinearSVC with TF-IDF features

Tuned hyperparameters include max_features, ngram_range, and C parameter for SVM

Labels include various bullying types such as age, religion, ethnicity, and others grouped under "Other Bullying"

Notes
The "Other Bullying" category includes gender, sexual orientation, appearance, disability, language, and other bullying types not classified under age, religion, or ethnicity.

Use the app responsibly — it is designed for educational/testing purposes and should not be used to promote any form of bullying.
