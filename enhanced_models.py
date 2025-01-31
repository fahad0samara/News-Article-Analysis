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

# Sample dataset
SAMPLE_DATA = {
    'text': [
        # Technology articles
        "Apple unveils new iPhone with revolutionary AI capabilities and enhanced camera system",
        "Microsoft releases Windows update with advanced security features and cloud integration",
        "Google announces breakthrough in quantum computing research",
        "Tesla's new electric vehicle features autonomous driving capabilities",
        "Amazon launches new cloud computing service for artificial intelligence",
        
        # Business articles
        "Stock market reaches record high as tech companies report strong earnings",
        "Federal Reserve announces interest rate decision impact on economy",
        "Goldman Sachs reports quarterly profits exceeding market expectations",
        "Oil prices surge amid global supply chain concerns",
        "Startup raises million in Series A funding round",
        
        # Sports articles
        "Manchester United wins dramatic match with last-minute goal",
        "NBA player breaks scoring record in championship game",
        "Tennis star advances to Wimbledon finals after intense match",
        "Olympic athlete sets new world record in track event",
        "Football team announces new coach for upcoming season",
        
        # Entertainment articles
        "New Marvel movie breaks box office records worldwide",
        "Popular singer announces world tour dates",
        "Netflix series wins multiple Emmy awards",
        "Celebrity couple announces engagement on social media",
        "New video game release attracts millions of players",
        
        # Politics articles
        "President signs new climate change legislation",
        "Parliament debates new economic policy",
        "Election results show shift in voter preferences",
        "International summit addresses global security concerns",
        "Political leader announces campaign for upcoming election"
    ],
    'category': [
        'tech', 'tech', 'tech', 'tech', 'tech',
        'business', 'business', 'business', 'business', 'business',
        'sports', 'sports', 'sports', 'sports', 'sports',
        'entertainment', 'entertainment', 'entertainment', 'entertainment', 'entertainment',
        'politics', 'politics', 'politics', 'politics', 'politics'
    ]
}

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
            voting='hard'  # Using hard voting since LinearSVC doesn't support predict_proba
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

def train_and_save_model(model_output_path):
    """Train model using sample data and save it"""
    # Create DataFrame from sample data
    df = pd.DataFrame(SAMPLE_DATA)
    
    # Initialize classifier
    classifier = NewsClassifier()
    
    # Train model
    results = classifier.train(df['text'], df['category'])
    
    # Save model
    classifier.save_model(model_output_path)
    
    return results

if __name__ == "__main__":
    # Train and save model
    model_output_path = "models/ensemble_model.joblib"
    results = train_and_save_model(model_output_path)
    print("\nClassification Report:")
    print(results['classification_report'])
