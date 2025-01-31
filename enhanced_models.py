import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

class NewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize base classifiers
        self.gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        self.svm = LinearSVC(
            C=1.0,
            random_state=42,
            dual=False
        )
        
        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('gb', self.gb),
                ('lr', self.lr),
                ('svm', self.svm)
            ],
            voting='hard'  # Changed to hard voting since LinearSVC doesn't support predict_proba
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.ensemble)
        ])

    def train(self, X, y):
        """Train the model on given data"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        
        # Generate reports
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'test_data': (X_test, y_test, y_pred)
        }

    def predict(self, text):
        """Predict category for given text"""
        return self.pipeline.predict([text])[0]

    def save_model(self, filepath):
        """Save model to file"""
        joblib.dump(self.pipeline, filepath)

    @staticmethod
    def load_model(filepath):
        """Load model from file"""
        return joblib.load(filepath)

def train_and_save_model(data_path, model_output_path):
    """Train model and save it to file"""
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize classifier
    classifier = NewsClassifier()
    
    # Train model
    results = classifier.train(df['text'], df['category'])
    
    # Save model
    classifier.save_model(model_output_path)
    
    return results

if __name__ == "__main__":
    # Example usage
    data_path = "bbc-text-cleaned.csv"  # Update with your data path
    model_output_path = "models/ensemble_model.joblib"
    
    results = train_and_save_model(data_path, model_output_path)
    print("\nClassification Report:")
    print(results['classification_report'])
