**Cyberbullying Tweet Classifier**

This project builds a machine learning model to detect and classify types of cyberbullying in tweets. The app lets users input tweet text and instantly get predictions about the type of cyberbullying present.

---

**Features**

- Cleans and preprocesses tweet text
- Uses TF-IDF vectorization and LinearSVC classifier
- Hyperparameter tuning with GridSearchCV
- Saves the best model for deployment
- Interactive, user-friendly Streamlit web app for classification
- Explains "Other Bullying" category for better user understanding

---

**Usage**

**Running the Streamlit App**

1. Make sure the saved model file (`best_cyberbullying_model.pkl`) is inside the `Model` folder.  
2. Run the Streamlit application by executing the following command in your terminal:


3. A new browser window/tab will open with the app interface.  
4. Enter a tweet in the text box and click the **Predict** button to see the predicted cyberbullying type.  
5. If the prediction is **Other Bullying**, click the expandable info box to learn more about what that includes.

---

**Training the Model (Optional)**

If you want to retrain the model from scratch or with updated data:

1. Ensure your dataset CSV file is placed correctly inside the `Data` folder (e.g., `cyberbullying_tweets.csv`).  
2. Run the training script:


3. The script will clean the data, perform hyperparameter tuning, train the model, and save the best model to the `Model` folder as `best_cyberbullying_model.pkl`.  
4. You can then run the Streamlit app again to use the updated model.

---

**File Structure**

- `train_model.py` — Script for training the model and saving it  
- `app.py` — Streamlit app code for the GUI  
- `Model/best_cyberbullying_model.pkl` — Saved trained model  
- `Data/cyberbullying_tweets.csv` — Dataset CSV file  
- `requirements.txt` — Required Python packages  

---

**Model Details**

- Model: LinearSVC with TF-IDF features  
- Tuned hyperparameters include max_features, ngram_range, and C parameter for SVM  
- Labels include various bullying types such as age, religion, ethnicity, and others grouped under "Other Bullying"

---

**Notes**

- The "Other Bullying" category includes gender, sexual orientation, appearance, disability, language, and other bullying types not classified under age, religion, or ethnicity.  
- Use the app responsibly — it is designed for educational/testing purposes and should not be used to promote any form of bullying.
