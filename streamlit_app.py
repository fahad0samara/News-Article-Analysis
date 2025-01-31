import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

# Sample dataset for training
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

@st.cache_resource
def train_model():
    """Train the model using sample data"""
    # Create DataFrame
    df = pd.DataFrame(SAMPLE_DATA)
    
    # Create pipeline
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Initialize classifiers
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    svm = LinearSVC(
        C=1.0,
        random_state=42,
        dual=False
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('gb', gb),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='hard'
    )
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', ensemble)
    ])
    
    pipeline.fit(df['text'], df['category'])
    return pipeline

def get_key_phrases(text, top_n=8):
    """Extract key phrases from text"""
    try:
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
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
    st.set_page_config(
        page_title="📰 News Article Classifier",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Article Classifier")
    st.write("This app classifies news articles into different categories using machine learning.")
    
    # Add sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses an ensemble of machine learning models to classify news articles into categories:
        - Technology 💻
        - Business 💼
        - Sports 🏆
        - Entertainment 🎬
        - Politics 🏛️
        """)
        
        st.header("Features")
        st.write("""
        - Multi-model ensemble classification
        - Key phrase extraction
        - Confidence scores
        - Real-time analysis
        """)
        
        st.header("How to Use")
        st.write("""
        1. Enter or paste your article text
        2. Click 'Classify Article'
        3. View results and analysis
        """)
    
    # Load/train model
    with st.spinner("Training model..."):
        model = train_model()
    
    # Text input
    article = st.text_area(
        "Enter your article text:",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    if st.button("Classify Article", type="primary"):
        if not article:
            st.warning("Please enter an article to classify.")
            return
            
        try:
            # Make prediction
            prediction = model.predict([article])[0]
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Result")
                
                # Show prediction with emoji
                category_emoji = {
                    'tech': '💻 Technology',
                    'business': '💼 Business',
                    'sports': '🏆 Sports',
                    'entertainment': '🎬 Entertainment',
                    'politics': '🏛️ Politics'
                }
                
                st.success(f"Category: {category_emoji.get(prediction, prediction)}")
                
                # Show sample articles
                st.subheader("Similar Articles")
                similar_articles = [text for text, cat in zip(SAMPLE_DATA['text'], SAMPLE_DATA['category']) 
                                 if cat == prediction][:2]
                for article in similar_articles:
                    st.info(article)
            
            with col2:
                # Extract and display key phrases
                st.subheader("Key Phrases")
                key_phrases = get_key_phrases(article)
                if key_phrases:
                    for phrase in key_phrases:
                        st.write(f"• {phrase}")
                else:
                    st.write("No key phrases extracted")
                
                # Show example categories
                st.subheader("Category Examples")
                st.write("""
                💻 Tech: AI, software, gadgets
                💼 Business: markets, economy, trade
                🏆 Sports: games, tournaments, athletes
                🎬 Entertainment: movies, music, celebrities
                🏛️ Politics: government, policy, elections
                """)
            
        except Exception as e:
            st.error(f"Error classifying article: {str(e)}")
            st.write("Please try again with a different article.")

if __name__ == "__main__":
    main()
