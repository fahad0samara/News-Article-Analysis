import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
import joblib
import re
from sklearn.base import BaseEstimator, TransformerMixin

class ContextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.contexts = {
            'tech': {
                'keywords': ['technology', 'tech', 'ai', 'software', 'hardware', 'digital', 'innovation', 
                           'smartphone', 'iphone', 'android', 'app', 'computer', 'artificial intelligence',
                           'machine learning', 'cloud', 'cybersecurity', '5g', 'blockchain', 'mobile',
                           'device', 'platform', 'algorithm', 'interface', 'processor', 'chip'],
                'companies': ['apple', 'google', 'microsoft', 'amazon', 'meta', 'tesla', 'nvidia', 
                            'samsung', 'intel', 'ibm', 'oracle', 'cisco', 'qualcomm', 'adobe']
            },
            'business': {
                'keywords': ['earnings', 'revenue', 'profit', 'market', 'stock', 'shares', 'investors',
                           'quarterly', 'financial', 'economy', 'growth', 'sales', 'trading', 'price',
                           'investment', 'dividend', 'merger', 'acquisition', 'fiscal', 'shareholder'],
                'terms': ['q1', 'q2', 'q3', 'q4', 'year-over-year', 'yoy', 'quarter', 'fiscal']
            },
            'sports': {
                'keywords': ['game', 'match', 'tournament', 'championship', 'league', 'score', 'win',
                           'victory', 'team', 'player', 'season', 'coach', 'stadium', 'sports'],
                'terms': ['goal', 'points', 'referee', 'injury', 'transfer', 'contract']
            },
            'entertainment': {
                'keywords': ['movie', 'film', 'show', 'music', 'album', 'celebrity', 'actor', 'actress',
                           'director', 'performance', 'award', 'entertainment', 'concert', 'premiere'],
                'terms': ['box office', 'rating', 'review', 'star', 'episode', 'season']
            },
            'politics': {
                'keywords': ['government', 'policy', 'election', 'political', 'minister', 'president',
                           'congress', 'senate', 'law', 'legislation', 'vote', 'campaign', 'party'],
                'terms': ['bill', 'reform', 'regulation', 'democratic', 'republican', 'parliament']
            }
        }

    def get_context_features(self, text):
        text_lower = text.lower()
        features = {}
        
        for context, indicators in self.contexts.items():
            # Keyword score
            keyword_matches = sum(1 for keyword in indicators['keywords'] 
                                if keyword in text_lower)
            features[f'{context}_keyword_score'] = keyword_matches
            
            # Special terms score
            if 'terms' in indicators:
                term_matches = sum(1 for term in indicators['terms'] 
                                 if term in text_lower)
                features[f'{context}_term_score'] = term_matches
            
            # Company/Entity score (for tech)
            if 'companies' in indicators:
                company_matches = sum(1 for company in indicators['companies'] 
                                    if company in text_lower)
                features[f'{context}_company_score'] = company_matches * 2  # Weight companies more
                
        return features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = []
        for text in X:
            features = self.get_context_features(text)
            features_list.append(features)
        return pd.DataFrame(features_list)

def train_enhanced_models():
    print("Loading data...")
    df = pd.read_csv('bbc-text-cleaned.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['category']
    )
    
    print("\nTraining enhanced models...")
    
    # Create feature union pipeline
    enhanced_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('context', ContextFeatureExtractor())
        ])),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5
        ))
    ])
    
    # Train enhanced gradient boosting
    enhanced_pipeline.fit(X_train, y_train)
    
    # Create and train enhanced ensemble
    enhanced_ensemble = VotingClassifier(estimators=[
        ('gb', enhanced_pipeline),
        ('lr', Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('context', ContextFeatureExtractor())
            ])),
            ('classifier', LogisticRegression(max_iter=1000))
        ])),
        ('svm', Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('context', ContextFeatureExtractor())
            ])),
            ('classifier', LinearSVC(max_iter=1000))
        ]))
    ], voting='hard')
    
    enhanced_ensemble.fit(X_train, y_train)
    
    # Save models
    print("\nSaving models...")
    joblib.dump(enhanced_pipeline, 'enhanced_models/gradient_boosting_model.joblib')
    joblib.dump(enhanced_ensemble, 'enhanced_models/ensemble_model.joblib')
    
    # Evaluate models
    print("\nEvaluating models...")
    with open('enhanced_results/model_evaluation.txt', 'w') as f:
        f.write("=== Enhanced Model Evaluation ===\n\n")
        
        for name, model in [('Gradient Boosting', enhanced_pipeline), 
                          ('Ensemble', enhanced_ensemble)]:
            accuracy = model.score(X_test, y_test)
            f.write(f"{name} Results:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
    
    print("Training complete! Models saved in 'enhanced_models' directory.")

if __name__ == "__main__":
    train_enhanced_models()
