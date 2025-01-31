import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from scipy import stats

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Read the CSV file
print("Loading and checking data quality...")
df = pd.read_csv('bbc-text-1.csv')

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing Percentage': missing_percentages
})
print(missing_info)

# Check for empty strings or whitespace
print("\nChecking for empty strings or whitespace:")
empty_strings = (df == '').sum()
whitespace_strings = df.apply(lambda x: (x.str.isspace() if x.dtype == 'object' else False).sum() if not x.empty else 0)
empty_info = pd.DataFrame({
    'Empty Strings': empty_strings,
    'Whitespace Only': whitespace_strings
})
print(empty_info)

# Data quality checks
print("\nData Quality Summary:")
print(f"Total rows: {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Save missing value analysis to file
with open('data_quality_report.txt', 'w') as f:
    f.write("=== BBC News Dataset Quality Report ===\n\n")
    
    f.write("1. Missing Values Analysis:\n")
    f.write(str(missing_info) + "\n\n")
    
    f.write("2. Empty/Whitespace Analysis:\n")
    f.write(str(empty_info) + "\n\n")
    
    f.write("3. Data Quality Summary:\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")
    
    # Category value counts
    f.write("4. Category Distribution:\n")
    f.write(str(df['category'].value_counts()) + "\n\n")
    
    # Text length statistics
    f.write("5. Text Length Statistics:\n")
    text_lengths = df['text'].str.len()
    f.write(f"Minimum text length: {text_lengths.min()}\n")
    f.write(f"Maximum text length: {text_lengths.max()}\n")
    f.write(f"Average text length: {text_lengths.mean():.2f}\n")
    
    # Check for unusual patterns
    short_texts = len(df[df['text'].str.len() < 100])
    long_texts = len(df[df['text'].str.len() > 10000])
    f.write(f"\nUnusual text lengths:\n")
    f.write(f"Texts shorter than 100 characters: {short_texts}\n")
    f.write(f"Texts longer than 10000 characters: {long_texts}\n")

print("\nData quality report has been saved to 'data_quality_report.txt'")

# Continue with the rest of the analysis if there are no critical issues
if df.isnull().sum().sum() == 0:
    print("\nNo missing values found, proceeding with analysis...")
    
    # Data Preprocessing
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    # Clean the text column
    df['clean_text'] = df['text'].apply(clean_text)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['clean_text'].apply(len)
    df['avg_word_length'] = df['clean_text'].apply(lambda x: np.mean([len(word) for word in x.split()]))

    # Label Encoding for correlation analysis
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_features = tfidf.fit_transform(df['clean_text'])
    top_words = pd.DataFrame(tfidf_features.toarray(), columns=tfidf.get_feature_names_out())

    # Save basic statistics
    with open('eda_results.txt', 'w') as f:
        f.write("=== BBC News Dataset EDA ===\n\n")
        
        # Basic Dataset Info
        f.write("1. Basic Dataset Information:\n")
        f.write(f"Total articles: {len(df)}\n")
        f.write(f"Categories: {', '.join(df['category'].unique())}\n\n")
        
        # Statistical Summary
        f.write("2. Statistical Summary of Text Features:\n")
        stats_summary = df[['word_count', 'char_count', 'avg_word_length']].describe()
        f.write(str(stats_summary) + "\n\n")
        
        # Category Distribution
        f.write("3. Category Distribution:\n")
        category_dist = df['category'].value_counts()
        f.write(str(category_dist) + "\n\n")
        
        # Text Length Statistics by Category
        f.write("4. Average Text Length by Category:\n")
        avg_length = df.groupby('category')['word_count'].mean().sort_values(ascending=False)
        f.write(str(avg_length) + "\n\n")

    # Create visualizations
    print("Generating visualizations...")

    # 1. Distribution Plots
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.histplot(data=df, x='word_count', bins=50)
    plt.title('Distribution of Word Count')
    plt.xlabel('Word Count')

    plt.subplot(132)
    sns.histplot(data=df, x='char_count', bins=50)
    plt.title('Distribution of Character Count')
    plt.xlabel('Character Count')

    plt.subplot(133)
    sns.histplot(data=df, x='avg_word_length', bins=50)
    plt.title('Distribution of Average Word Length')
    plt.xlabel('Average Word Length')
    plt.tight_layout()
    plt.savefig('distribution_plots.png')
    plt.close()

    # 2. Box Plots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='word_count')
    plt.title('Word Count Distribution by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_boxplots.png')
    plt.close()

    # 3. Correlation Heatmap
    numeric_cols = ['word_count', 'char_count', 'avg_word_length', 'category_encoded']
    correlation_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 4. Word Clouds for each category
    for category in df['category'].unique():
        text = ' '.join(df[df['category'] == category]['clean_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {category.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'wordcloud_{category}.png')
        plt.close()

    # 5. Statistical Tests
    # Perform Shapiro-Wilk test for normality on word_count
    _, p_value = stats.shapiro(df['word_count'])
    with open('eda_results.txt', 'a') as f:
        f.write("\n5. Statistical Tests:\n")
        f.write(f"Shapiro-Wilk test p-value for word_count: {p_value}\n")
        f.write("(p-value < 0.05 indicates non-normal distribution)\n")

    print("Analysis complete! Check eda_results.txt and the generated plots.")

else:
    print("\nWarning: Dataset contains missing values. Please handle them before proceeding with analysis.")

print("Data quality analysis complete! Check data_quality_report.txt for details.")
