import streamlit as st
import joblib
import pandas as pd

st.title('News Article Classifier')

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = ['gradient_boosting_model.joblib', 'ensemble_model.joblib']
    for model_file in model_files:
        name = model_file.replace('_model.joblib', '')
        models[name] = joblib.load(f'models/{model_file}')
    return models

models = load_models()

# Create interface
st.write('Enter a news article to classify:')
text = st.text_area('Article text:', height=200)
model_name = st.selectbox('Select model:', list(models.keys()))

if st.button('Classify'):
    if text:
        # Make prediction
        model = models[model_name]
        prediction = model.predict([text])[0]
        
        # Show result
        st.write('Predicted category:', prediction)
        
        # Show confidence scores if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba([text])[0]
            st.write('\nConfidence scores:')
            for category, prob in zip(model.classes_, probs):
                st.write(f'{category}: {prob:.4f}')
    else:
        st.write('Please enter some text to classify.')
