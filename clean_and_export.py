import pandas as pd
import numpy as np

# Read the original dataset
print("Reading original dataset...")
df = pd.read_csv('bbc-text-1.csv')

print(f"Original dataset shape: {df.shape}")

# Remove duplicates
print("Removing duplicates...")
df_cleaned = df.drop_duplicates(subset=['text'], keep='first')
print(f"Shape after removing duplicates: {df_cleaned.shape}")

# Clean text: remove extra whitespace and standardize
print("Cleaning text...")
df_cleaned['text'] = df_cleaned['text'].str.strip()

# Remove articles that are too long (outliers)
long_text_threshold = df_cleaned['text'].str.len().quantile(0.99)  # 99th percentile
df_cleaned = df_cleaned[df_cleaned['text'].str.len() <= long_text_threshold]
print(f"Shape after removing very long articles: {df_cleaned.shape}")

# Add some useful metadata columns
df_cleaned['text_length'] = df_cleaned['text'].str.len()
df_cleaned['word_count'] = df_cleaned['text'].str.split().str.len()

# Sort by category and text length
df_cleaned = df_cleaned.sort_values(['category', 'text_length'])

# Export to new CSV file
output_file = 'bbc-text-cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned dataset exported to {output_file}")

# Print summary of changes
print("\nCleaning Summary:")
print(f"Original number of articles: {len(df)}")
print(f"Final number of articles: {len(df_cleaned)}")
print(f"Total rows removed: {len(df) - len(df_cleaned)}")

# Save detailed report
with open('cleaning_report.txt', 'w') as f:
    f.write("=== Data Cleaning Report ===\n\n")
    f.write(f"Original dataset size: {len(df)} articles\n")
    f.write(f"Cleaned dataset size: {len(df_cleaned)} articles\n")
    f.write(f"Rows removed: {len(df) - len(df_cleaned)}\n\n")
    
    f.write("Category distribution in cleaned dataset:\n")
    f.write(str(df_cleaned['category'].value_counts()) + "\n\n")
    
    f.write("Text length statistics in cleaned dataset:\n")
    f.write(str(df_cleaned['text_length'].describe()) + "\n\n")
    
    f.write("Word count statistics in cleaned dataset:\n")
    f.write(str(df_cleaned['word_count'].describe()))
