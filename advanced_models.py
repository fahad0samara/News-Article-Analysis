import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st
import os

# Create directories
for dir_name in ['advanced_ml_results', 'models']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class NewsClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = None
        
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
    
    def train_advanced_models(self, X_train, X_test, y_train, y_test):
        print("\nTraining advanced models...")
        
        # 1. Gradient Boosting with GridSearch
        gb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', GradientBoostingClassifier())
        ])
        
        gb_params = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
        
        gb_grid = GridSearchCV(gb_pipeline, gb_params, cv=3, n_jobs=-1)
        gb_grid.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_grid.best_estimator_
        
        # 2. Ensemble of best models
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        
        ensemble = VotingClassifier(estimators=[
            ('gb', gb_grid.best_estimator_),
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
            joblib.dump(model, f'models/{name}_model.joblib')
        
        # Evaluate models
        results = {}
        with open('advanced_ml_results/advanced_evaluation.txt', 'w') as f:
            f.write("=== Advanced Model Evaluation ===\n\n")
            
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                f.write(f"{name.replace('_', ' ').title()} Results:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred, 
                    target_names=self.label_encoder.classes_) + "\n\n")
        
        return results

    def create_visualizations(self, results, y_test):
        print("\nGenerating visualizations...")
        
        # 1. Model Comparison
        plt.figure(figsize=(10, 6))
        accuracies = [results[name]['accuracy'] for name in self.models.keys()]
        plt.bar(range(len(accuracies)), accuracies)
        plt.xticks(range(len(accuracies)), 
                  [name.replace('_', ' ').title() for name in self.models.keys()],
                  rotation=45)
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        plt.savefig('advanced_ml_results/model_comparison.png')
        plt.close()
        
        # 2. Confusion Matrices
        for name, result in results.items():
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix - {name.replace("_", " ").title()}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'advanced_ml_results/confusion_matrix_{name}.png')
            plt.close()

def create_prediction_interface():
    """Create a Streamlit interface for predictions"""
    with open('predict_interface.py', 'w') as f:
        f.write("""import streamlit as st
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
            st.write('\\nConfidence scores:')
            for category, prob in zip(model.classes_, probs):
                st.write(f'{category}: {prob:.4f}')
    else:
        st.write('Please enter some text to classify.')
""")

if __name__ == "__main__":
    # Train and evaluate models
    classifier = NewsClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    results = classifier.train_advanced_models(X_train, X_test, y_train, y_test)
    classifier.create_visualizations(results, y_test)
    
    # Create prediction interface
    create_prediction_interface()
    
    print("\nAdvanced analysis complete! To use the prediction interface, run:")
    print("streamlit run predict_interface.py")
