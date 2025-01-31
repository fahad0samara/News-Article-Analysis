import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib
import nltk
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_model():
    try:
        model = joblib.load('models/ensemble_model.joblib')
        return model
    except:
        st.error("Error loading model. Please ensure model file exists in the models directory.")
        return None

def get_key_phrases(text, lang='en', top_n=8):
    try:
        # Detect language if not specified
        if lang == 'auto':
            lang = detect(text)
        
        # Get appropriate stopwords
        stop_words = set(stopwords.words('english' if lang == 'en' else 'french'))
        
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get word frequencies
        freq_dist = nltk.FreqDist(words)
        
        # Get bigrams
        bigrams = list(nltk.bigrams(words))
        bigram_freq = nltk.FreqDist(bigrams)
        
        # Combine single words and bigrams
        phrases = [(word, freq) for word, freq in freq_dist.items()]
        phrases.extend([(' '.join(bigram), freq) for bigram, freq in bigram_freq.items()])
        
        # Sort by frequency and get top phrases
        top_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)[:top_n]
        return [phrase[0] for phrase in top_phrases]
    except Exception as e:
        st.error(f"Error extracting key phrases: {str(e)}")
        return []

def main():
    st.title("📰 News Article Classifier")
    st.write("Enter your news article below to classify it into categories.")

    # Load model
    model = load_model()
    if model is None:
        return

    # Text input
    article = st.text_area("Article Text", height=200)
    
    if st.button("Classify"):
        if not article:
            st.warning("Please enter an article to classify.")
            return
            
        try:
            # Make prediction
            prediction = model.predict([article])[0]
            probabilities = model.predict_proba([article])[0]
            
            # Get confidence score
            confidence = max(probabilities) * 100
            
            # Get key phrases
            key_phrases = get_key_phrases(article)
            
            # Display results
            st.success(f"Category: {prediction}")
            st.info(f"Confidence: {confidence:.2f}%")
            
            # Display probabilities for all classes
            st.subheader("Category Probabilities")
            proba_df = pd.DataFrame({
                'Category': model.classes_,
                'Probability': probabilities * 100
            })
            st.write(proba_df)
            
            # Display key phrases
            if key_phrases:
                st.subheader("Key Phrases")
                st.write(", ".join(key_phrases))
                
        except Exception as e:
            st.error(f"Error classifying article: {str(e)}")

if __name__ == "__main__":
    main()
