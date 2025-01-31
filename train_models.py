import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create directory for ML results
if not os.path.exists('ml_results'):
    os.makedirs('ml_results')

# Load the cleaned dataset
print("Loading data...")
df = pd.read_csv('bbc-text-cleaned.csv')

# Split features and target
X = df['text']
y = df['category']

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create model pipelines
models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', MultinomialNB())
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', LinearSVC(max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', RandomForestClassifier())
    ])
}

# Train and evaluate models
results = {}
print("\nTraining and evaluating models...")

with open('ml_results/model_evaluation.txt', 'w') as f:
    f.write("=== Model Evaluation Results ===\n\n")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate scores
        accuracy = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
        
        # Save model
        joblib.dump(model, f'ml_results/{name.lower().replace(" ", "_")}_model.joblib')
        
        # Write results to file
        f.write(f"{name} Results:\n")
        f.write("-" * (len(name) + 9) + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Cross-validation Scores: {cv_scores}\n")
        f.write(f"Average CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n\n")

# Visualize results
print("\nGenerating visualizations...")

# 1. Model Comparison Plot
plt.figure(figsize=(10, 6))
accuracies = [results[name]['accuracy'] for name in models.keys()]
cv_means = [results[name]['cv_scores'].mean() for name in models.keys()]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
plt.bar(x + width/2, cv_means, width, label='CV Average')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, models.keys(), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('ml_results/model_comparison.png')
plt.close()

# 2. Confusion Matrices
for name, result in results.items():
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(df['category'].unique()),
                yticklabels=sorted(df['category'].unique()))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'ml_results/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Function to make predictions
def predict_category(text, model_name='Logistic Regression'):
    model = joblib.load(f'ml_results/{model_name.lower().replace(" ", "_")}_model.joblib')
    prediction = model.predict([text])[0]
    probabilities = None
    
    # Get probability scores if model supports it
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([text])[0]
    
    return prediction, probabilities

# Save example predictions
print("\nGenerating example predictions...")
example_texts = [
    "The stock market showed significant gains today as tech companies reported strong earnings.",
    "The football match ended in a dramatic penalty shootout.",
    "The new smartphone features an innovative camera system and improved battery life.",
    "The Oscar-winning actor announced their next big movie project.",
    "The prime minister addressed concerns about the new policy in parliament."
]

with open('ml_results/example_predictions.txt', 'w') as f:
    f.write("=== Example Predictions ===\n\n")
    for text in example_texts:
        f.write(f"Text: {text}\n")
        for model_name in models.keys():
            prediction, _ = predict_category(text, model_name)
            f.write(f"{model_name}: {prediction}\n")
        f.write("\n")

print("\nMachine Learning analysis complete! Check the 'ml_results' directory for all outputs.")

# Print best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest performing model: {best_model[0]}")
print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
