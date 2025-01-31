import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

print("Loading cleaned dataset...")
df = pd.read_csv('bbc-text-cleaned.csv')

# Create analysis directory if it doesn't exist
import os
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')

# 1. Category Analysis
print("\nAnalyzing categories...")
plt.figure(figsize=(10, 6))
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Distribution of Articles by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_results/category_distribution.png')
plt.close()

# 2. Text Length Analysis
print("Analyzing text lengths...")
plt.figure(figsize=(15, 5))

plt.subplot(131)
sns.histplot(data=df, x='text_length', bins=50)
plt.title('Distribution of Text Lengths')

plt.subplot(132)
sns.boxplot(data=df, x='category', y='text_length')
plt.title('Text Length by Category')
plt.xticks(rotation=45)

plt.subplot(133)
sns.violinplot(data=df, x='category', y='word_count')
plt.title('Word Count Distribution by Category')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('analysis_results/length_analysis.png')
plt.close()

# 3. Most Common Words by Category
print("Analyzing common words by category...")
stop_words = set(stopwords.words('english'))

def get_top_words(text_series, n=10):
    words = ' '.join(text_series).lower().split()
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(n)

category_top_words = {}
for category in df['category'].unique():
    category_top_words[category] = get_top_words(df[df['category'] == category]['text'])

# Create word clouds for each category
print("Generating word clouds...")
for category in df['category'].unique():
    text = ' '.join(df[df['category'] == category]['text'])
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         stopwords=stop_words).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {category.capitalize()}')
    plt.tight_layout()
    plt.savefig(f'analysis_results/wordcloud_{category}.png')
    plt.close()

# 4. Text Complexity Analysis
print("Analyzing text complexity...")
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
df['sentence_count'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(str(x))))
df['avg_sentence_length'] = df['word_count'] / df['sentence_count']

# Complexity visualization
plt.figure(figsize=(15, 5))

plt.subplot(131)
sns.boxplot(data=df, x='category', y='avg_word_length')
plt.title('Average Word Length by Category')
plt.xticks(rotation=45)

plt.subplot(132)
sns.boxplot(data=df, x='category', y='sentence_count')
plt.title('Sentence Count by Category')
plt.xticks(rotation=45)

plt.subplot(133)
sns.boxplot(data=df, x='category', y='avg_sentence_length')
plt.title('Average Sentence Length by Category')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('analysis_results/complexity_analysis.png')
plt.close()

# Save detailed analysis report
print("\nSaving analysis report...")
with open('analysis_results/detailed_analysis.txt', 'w') as f:
    f.write("=== BBC News Cleaned Dataset Analysis ===\n\n")
    
    f.write("1. Basic Statistics\n")
    f.write("-----------------\n")
    f.write(f"Total articles: {len(df)}\n")
    f.write("\nCategory Distribution:\n")
    f.write(str(df['category'].value_counts()) + "\n\n")
    
    f.write("2. Text Length Statistics\n")
    f.write("---------------------\n")
    f.write("Word Count Statistics:\n")
    f.write(str(df['word_count'].describe()) + "\n\n")
    f.write("Text Length Statistics:\n")
    f.write(str(df['text_length'].describe()) + "\n\n")
    
    f.write("3. Text Complexity Metrics\n")
    f.write("----------------------\n")
    f.write("Average Word Length by Category:\n")
    f.write(str(df.groupby('category')['avg_word_length'].mean()) + "\n\n")
    f.write("Average Sentence Length by Category:\n")
    f.write(str(df.groupby('category')['avg_sentence_length'].mean()) + "\n\n")
    
    f.write("4. Most Common Words by Category\n")
    f.write("---------------------------\n")
    for category, words in category_top_words.items():
        f.write(f"\n{category.capitalize()}:\n")
        for word, count in words:
            f.write(f"{word}: {count}\n")

print("\nAnalysis complete! Check the 'analysis_results' directory for all outputs.")
