import os
import streamlit as st
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
model_path = os.path.join(MODEL_DIR, "best_cyberbullying_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Mapping for friendly labels
label_mapping = {
    "other_cyberbullying": "May include gender, sexual orientation, appearance, disability, language, or other types of bullying"
    # You can add other mappings here if you want
}

st.set_page_config(page_title="Cyberbullying Tweet Classifier", page_icon="🛡️", layout="centered")

st.title("🛡️ Cyberbullying Tweet Classifier")
st.markdown("""
Detect if a tweet contains cyberbullying content and identify the type.  
Enter your tweet below and get an instant prediction.
""")

tweet = st.text_area("Enter Tweet Text:", height=150, placeholder="Type or paste your tweet here...")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter some tweet text to classify.")
    else:
        prediction = model.predict([tweet])[0]
        friendly_label = label_mapping.get(prediction, prediction)
        st.success(f"**Predicted Cyberbullying Type:** {friendly_label}")

        if prediction == "other_cyberbullying":
            with st.expander("What does 'Other Bullying' mean?"):
                st.write("""
                The category **Other Bullying** may include:
                - Gender bullying  
                - Sexual orientation bullying (LGBTQ+ related)  
                - Appearance / Body image bullying (looks, weight, height)  
                - Disability bullying (physical or mental disabilities)  
                - Language bullying (accent, how someone speaks)  
                - Threats or harassment not classified elsewhere  
                - Body shaming  
                - Insults or name-calling unrelated to age, religion, or ethnicity  
                - Any other cyberbullying types outside age, religion, ethnicity
                """)
