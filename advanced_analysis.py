import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

print("Loading cleaned dataset...")
df = pd.read_csv('bbc-text-cleaned.csv')

# Create directory for advanced analysis
import os
if not os.path.exists('advanced_analysis'):
    os.makedirs('advanced_analysis')

# 1. Sentiment Analysis
print("\nPerforming sentiment analysis...")
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['sentiment'] = df['text'].apply(get_sentiment)

# Sentiment visualization
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='category', y='sentiment')
plt.title('Sentiment Distribution by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('advanced_analysis/sentiment_analysis.png')
plt.close()

# 2. Topic Modeling
print("Performing topic modeling...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
text_features = tfidf.fit_transform(df['text'])

# Create topic model
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
topic_results = lda.fit_transform(text_features)

# Get top words for each topic
feature_names = tfidf.get_feature_names_out()
top_words_per_topic = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
    top_words_per_topic.append(top_words)

# 3. Advanced Text Statistics
print("Calculating advanced text statistics...")

# Parts of speech analysis
def get_pos_tags(text):
    tokens = word_tokenize(str(text))
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags

# Calculate POS statistics for a sample of articles
sample_size = min(100, len(df))
sampled_indices = np.random.choice(len(df), sample_size, replace=False)
pos_stats = []

for idx in sampled_indices:
    text = df.iloc[idx]['text']
    pos_tags = get_pos_tags(text)
    pos_counts = pd.Series([tag for word, tag in pos_tags]).value_counts()
    pos_stats.append(pos_counts)

pos_df = pd.DataFrame(pos_stats).fillna(0)
pos_proportions = pos_df.mean()

# 4. Statistical Tests
print("Performing statistical tests...")

# ANOVA test for word count differences between categories
categories = df['category'].unique()
word_counts_by_category = [df[df['category'] == cat]['word_count'] for cat in categories]
f_stat, p_value = stats.f_oneway(*word_counts_by_category)

# Chi-square test for category distribution
observed_freq = df['category'].value_counts()
expected_freq = np.ones_like(observed_freq) * len(df) / len(categories)
chi2_stat, chi2_p = stats.chisquare(observed_freq, expected_freq)

# Save results
print("\nSaving advanced analysis results...")
with open('advanced_analysis/advanced_analysis_report.txt', 'w') as f:
    f.write("=== Advanced Analysis Report ===\n\n")
    
    f.write("1. Sentiment Analysis\n")
    f.write("-----------------\n")
    f.write("Average Sentiment by Category:\n")
    f.write(str(df.groupby('category')['sentiment'].mean()) + "\n\n")
    
    f.write("2. Topic Modeling Results\n")
    f.write("---------------------\n")
    for i, top_words in enumerate(top_words_per_topic):
        f.write(f"Topic {i+1}: {', '.join(top_words)}\n")
    f.write("\n")
    
    f.write("3. Parts of Speech Analysis\n")
    f.write("------------------------\n")
    f.write("Average POS Distribution:\n")
    f.write(str(pos_proportions.sort_values(ascending=False)) + "\n\n")
    
    f.write("4. Statistical Tests\n")
    f.write("----------------\n")
    f.write(f"ANOVA test for word count differences:\n")
    f.write(f"F-statistic: {f_stat:.4f}\n")
    f.write(f"p-value: {p_value:.4f}\n\n")
    f.write(f"Chi-square test for category distribution:\n")
    f.write(f"Chi-square statistic: {chi2_stat:.4f}\n")
    f.write(f"p-value: {chi2_p:.4f}\n\n")

# 5. Advanced Visualizations
print("Creating advanced visualizations...")

# Sentiment distribution
plt.figure(figsize=(15, 5))
plt.subplot(131)
sns.histplot(data=df, x='sentiment', bins=50)
plt.title('Overall Sentiment Distribution')

plt.subplot(132)
sns.violinplot(data=df, x='category', y='sentiment')
plt.title('Sentiment Distribution by Category')
plt.xticks(rotation=45)

plt.subplot(133)
sns.scatterplot(data=df, x='word_count', y='sentiment', hue='category', alpha=0.5)
plt.title('Sentiment vs. Word Count')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('advanced_analysis/sentiment_analysis_detailed.png')
plt.close()

# Topic distribution
topic_names = [f'Topic {i+1}' for i in range(n_topics)]
topic_probs = pd.DataFrame(topic_results, columns=topic_names)

plt.figure(figsize=(12, 6))
sns.boxplot(data=topic_probs)
plt.title('Distribution of Topic Probabilities')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('advanced_analysis/topic_distribution.png')
plt.close()

print("\nAdvanced analysis complete! Check the 'advanced_analysis' directory for results.")
