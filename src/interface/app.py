import streamlit as st
import joblib
import pandas as pd
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import re

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('stopwords', language='french')
except:
    pass

@st.cache_resource
def load_models():
    try:
        # Load the base models
        models = {}
        model_files = ['gradient_boosting_model.joblib', 'ensemble_model.joblib']
        category_names = ['business', 'entertainment', 'politics', 'sport', 'tech']
        
        for model_file in model_files:
            name = model_file.replace('_model.joblib', '')
            model = joblib.load(f'enhanced_models/{model_file}')
            
            # For pipeline models, get the classifier
            if hasattr(model, 'steps'):
                classifier = model.steps[-1][1]  # Get the last step (classifier)
                if not hasattr(classifier, 'classes_') or not isinstance(classifier.classes_[0], str):
                    classifier.classes_ = category_names
            else:
                if not hasattr(model, 'classes_') or not isinstance(model.classes_[0], str):
                    model.classes_ = category_names
            
            models[name] = model
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def clean_and_validate_text(text):
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text

def get_key_phrases(text, lang='en', top_n=5):
    try:
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        
        # Use appropriate stopwords based on language
        try:
            if lang == 'fr':
                stop_words = set(stopwords.words('french'))
            else:
                stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        # Add custom stopwords for better phrase extraction
        custom_stops = {'said', 'says', 'will', 'would', 'could', 'may', 'might', 'also', 'one', 'two', 'three', 'new'}
        stop_words.update(custom_stops)
        
        # Extract bigrams and single words
        words = [word for word in tokens if re.match(r'^[a-zA-ZÀ-ÿ]+$', word) and word not in stop_words]
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        # Combine single words and bigrams
        all_phrases = words + bigrams
        
        # Get frequency distribution
        freq_dist = nltk.FreqDist(all_phrases)
        
        # Filter out single-character words and get top phrases
        phrases = [(phrase, count) for phrase, count in freq_dist.most_common(top_n * 2)
                  if len(phrase) > 1][:top_n]
        
        return phrases
    except:
        return []

def get_category_name(category_id, model):
    """Convert numeric category to human-readable name"""
    try:
        # For pipeline models, get the classifier
        if hasattr(model, 'steps'):
            classifier = model.steps[-1][1]  # Get the last step (classifier)
        else:
            classifier = model
            
        if hasattr(classifier, 'classes_'):
            categories = classifier.classes_
            if isinstance(category_id, (int, np.integer)):
                category = categories[category_id]
            else:
                category = str(category_id)
            
            # Map categories to display names
            category_display = {
                'business': 'Business',
                'entertainment': 'Entertainment',
                'politics': 'Politics',
                'sport': 'Sports',
                'tech': 'Technology'
            }
            
            return category_display.get(category.lower(), category.title())
    except:
        pass
    return str(category_id)

def analyze_context(text):
    """Analyze the context of the article to help with classification"""
    text_lower = text.lower()
    
    # Define context indicators
    contexts = {
        'tech': {
            'keywords': ['technology', 'tech', 'ai', 'software', 'hardware', 'digital', 'innovation', 
                        'smartphone', 'iphone', 'android', 'app', 'computer', 'artificial intelligence',
                        'machine learning', 'cloud', 'cybersecurity', '5g', 'blockchain'],
            'companies': ['apple', 'google', 'microsoft', 'amazon', 'meta', 'tesla', 'nvidia', 
                         'samsung', 'intel', 'ibm', 'oracle']
        },
        'business': {
            'keywords': ['earnings', 'revenue', 'profit', 'market', 'stock', 'shares', 'investors',
                        'quarterly', 'financial', 'economy', 'growth', 'sales', 'trading', 'price',
                        'investment', 'dividend', 'merger', 'acquisition'],
            'terms': ['q1', 'q2', 'q3', 'q4', 'year-over-year', 'yoy', 'quarter', 'fiscal']
        }
    }
    
    # Count matches
    context_scores = {
        'tech': 0,
        'business': 0
    }
    
    # Check tech context
    for keyword in contexts['tech']['keywords']:
        if keyword in text_lower:
            context_scores['tech'] += 1
    for company in contexts['tech']['companies']:
        if company in text_lower:
            context_scores['tech'] += 2  # Weight company mentions more heavily
            
    # Check business context
    for keyword in contexts['business']['keywords']:
        if keyword in text_lower:
            context_scores['business'] += 1
    for term in contexts['business']['terms']:
        if term in text_lower:
            context_scores['business'] += 1
            
    return context_scores

# Page config
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title('📰 News Article Classifier')
st.markdown("""
This tool classifies news articles into different categories:
- Business
- Technology
- Politics
- Sports
- Entertainment

Features:
- Multiple classification models
- Key phrase extraction
- Confidence scores
""")

# Sidebar for model selection
with st.sidebar:
    st.header("Model Settings")
    model_name = st.selectbox(
        'Select Classification Model:',
        ['ensemble', 'gradient_boosting'],
        help="Choose the model to use for classification"
    )

# Main content
st.header("Article Input")
text = st.text_area(
    'Enter your news article:',
    height=200,
    help="Paste your article here. The model works best with English text.",
    placeholder="Paste your news article here..."
)

# Load models
models = load_models()

if st.button('Analyze Article', type='primary'):
    if not text or len(text.strip()) < 10:
        st.error("Please enter a valid article (at least 10 characters).")
    elif not models:
        st.error("Models failed to load. Please try again later.")
    else:
        try:
            with st.spinner('Analyzing article...'):
                # Clean text
                text = clean_and_validate_text(text)
                
                # Detect language
                try:
                    lang = detect(text)
                    if lang != 'en':
                        st.info(f"Detected language: {lang}. Note: The model works best with English text.")
                except:
                    lang = 'en'
                
                # Make prediction
                model = models[model_name]
                prediction = model.predict([text])[0]
                category_name = get_category_name(prediction, model)
                
                # Analyze context
                context_scores = analyze_context(text)
                
                # Display results
                st.subheader("Classification Results:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Category", category_name)
                    
                    # Show confidence scores if available
                    if hasattr(model, 'predict_proba'):
                        st.subheader("Confidence Scores:")
                        probs = model.predict_proba([text])[0]
                        
                        # Get classifier for proper category names
                        if hasattr(model, 'steps'):
                            classifier = model.steps[-1][1]
                        else:
                            classifier = model
                            
                        categories = [get_category_name(cat, model) for cat in classifier.classes_]
                        
                        # Sort by probability
                        category_probs = list(zip(categories, probs))
                        category_probs.sort(key=lambda x: x[1], reverse=True)
                        
                        for category, prob in category_probs:
                            st.progress(prob)
                            st.write(f"{category}: {prob:.2%}")
                
                with col2:
                    # Extract and display key phrases
                    st.subheader("Key Phrases:")
                    key_phrases = get_key_phrases(text, lang='en', top_n=8)  # Increased to 8 phrases
                    if key_phrases:
                        for phrase, count in key_phrases:
                            st.write(f"- {phrase} ({count} occurrences)")
                    else:
                        st.write("No key phrases extracted")
                    
                    # Show context analysis
                    if context_scores['tech'] > 0 or context_scores['business'] > 0:
                        st.subheader("Context Analysis:")
                        st.write("Detected contextual indicators:")
                        if context_scores['tech'] > 0:
                            st.write(f"- Technology context score: {context_scores['tech']}")
                        if context_scores['business'] > 0:
                            st.write(f"- Business context score: {context_scores['business']}")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Please try again with a different article or model.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Powered by Advanced ML Models</p>
</div>
""", unsafe_allow_html=True)
