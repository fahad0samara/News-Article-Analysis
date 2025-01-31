import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
import joblib
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer
import torch
import streamlit as st
from langdetect import detect
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class EnhancedNewsClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.label_encoder = None
        self.translator = None
        self.topic_model = None
        self.dictionary = None
        
        # Create directories
        for dir_name in ['enhanced_results', 'enhanced_models']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    def initialize_translation(self):
        print("Initializing translation model...")
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
    
    def translate_to_english(self, text, source_lang=None):
        if not source_lang:
            try:
                source_lang = detect(text)
            except:
                return text
        
        if source_lang == 'en':
            return text
        
        try:
            translation = self.translator(text, max_length=512)[0]['translation_text']
            return translation
        except:
            return text
    
    def prepare_data(self):
        print("Loading and preparing data...")
        df = pd.read_csv('bbc-text-cleaned.csv')
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['category'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def train_topic_model(self, texts, num_topics=5):
        print("Training topic model...")
        # Tokenize and prepare documents
        texts = [word_tokenize(text.lower()) for text in texts]
        stop_words = set(stopwords.words('english'))
        texts = [[word for word in doc if word not in stop_words] for doc in texts]
        
        # Create dictionary and corpus
        self.dictionary = Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # Train LDA model
        self.topic_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10
        )
    
    def extract_topics(self, text):
        if not self.topic_model or not self.dictionary:
            return None
        
        # Prepare text
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Get topic distribution
        bow = self.dictionary.doc2bow(tokens)
        return self.topic_model.get_document_topics(bow)
    
    def train_enhanced_models(self, X_train, X_test, y_train, y_test):
        print("\nTraining enhanced models...")
        
        # Train topic model
        self.train_topic_model(X_train)
        
        # 1. Enhanced Gradient Boosting
        gb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5
            ))
        ])
        
        gb_pipeline.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_pipeline
        
        # 2. Enhanced Ensemble
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        
        ensemble = VotingClassifier(estimators=[
            ('gb', gb_pipeline),
            ('lr', Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', LogisticRegression(max_iter=1000))
            ])),
            ('svm', Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', LinearSVC(max_iter=1000))
            ]))
        ], voting='hard')
        
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'enhanced_models/{name}_model.joblib')
        
        # Evaluate and analyze errors
        results = {}
        with open('enhanced_results/enhanced_evaluation.txt', 'w') as f:
            f.write("=== Enhanced Model Evaluation ===\n\n")
            
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                # Analyze errors
                errors = pd.Series(X_test)[y_pred != y_test]
                true_labels = pd.Series(y_test)[y_pred != y_test]
                pred_labels = pd.Series(y_pred)[y_pred != y_test]
                
                f.write(f"{name.replace('_', ' ').title()} Results:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred,
                    target_names=self.label_encoder.classes_) + "\n\n")
                
                f.write("Error Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total errors: {len(errors)}\n\n")
                
                # Sample of misclassified examples
                f.write("Sample of misclassified examples:\n")
                for i in range(min(5, len(errors))):
                    f.write(f"\nExample {i+1}:\n")
                    f.write(f"Text: {errors.iloc[i][:200]}...\n")
                    f.write(f"True category: {self.label_encoder.classes_[true_labels.iloc[i]]}\n")
                    f.write(f"Predicted category: {self.label_encoder.classes_[pred_labels.iloc[i]]}\n")
                f.write("\n")
        
        return results

def create_enhanced_interface():
    """Create an enhanced Streamlit interface with multi-language support and topic analysis"""
    with open('enhanced_interface.py', 'w') as f:
        f.write("""import streamlit as st
import joblib
import pandas as pd
from langdetect import detect
from transformers import pipeline
import nltk
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Initialize translation
@st.cache_resource
def load_translator():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

@st.cache_resource
def load_models():
    models = {}
    model_files = ['gradient_boosting_model.joblib', 'ensemble_model.joblib']
    for model_file in model_files:
        name = model_file.replace('_model.joblib', '')
        models[name] = joblib.load(f'enhanced_models/{model_file}')
    return models

st.title('Enhanced News Article Classifier')

# Load models
translator = load_translator()
models = load_models()

# Create interface
st.write('Enter a news article in any language to classify:')
text = st.text_area('Article text:', height=200)
model_name = st.selectbox('Select model:', list(models.keys()))

if st.button('Analyze'):
    if text:
        # Detect language and translate if needed
        try:
            lang = detect(text)
            st.write(f'Detected language: {lang}')
            
            if lang != 'en':
                translation = translator(text, max_length=512)[0]['translation_text']
                st.write('Translated text:')
                st.write(translation)
                text = translation
        except:
            st.write('Language detection failed, proceeding with original text')
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict([text])[0]
        
        # Show results
        st.write('\\nResults:')
        st.write('Predicted category:', prediction)
        
        # Show confidence scores if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba([text])[0]
            st.write('\\nConfidence scores:')
            for category, prob in zip(model.classes_, probs):
                st.write(f'{category}: {prob:.4f}')
        
        # Extract key phrases
        try:
            doc = nltk.word_tokenize(text.lower())
            freq_dist = nltk.FreqDist(doc)
            st.write('\\nKey phrases:')
            for word, freq in freq_dist.most_common(5):
                st.write(f'{word}: {freq}')
        except:
            st.write('Could not extract key phrases')
            
    else:
        st.write('Please enter some text to classify.')
""")

if __name__ == "__main__":
    # Initialize and train enhanced models
    classifier = EnhancedNewsClassifier()
    classifier.initialize_translation()
    
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    results = classifier.train_enhanced_models(X_train, X_test, y_train, y_test)
    
    # Create enhanced interface
    create_enhanced_interface()
    
    print("\nEnhanced analysis complete! To use the enhanced interface, run:")
    print("streamlit run enhanced_interface.py")
