import pandas as pd
import re
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text):
    """
    Enhanced text cleaning for resume data
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but keep important punctuation for skills
    text = re.sub(r'[^a-zA-Z0-9\s.,!?()\-\+&/]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text.strip()

# Load data
print("Loading data...")
df = pd.read_csv("data/UpdatedResumeDataSet.csv")

# Check data
print(f"Total samples: {len(df)}")
print(f"Categories: {df['Category'].unique()}")
print(f"Class distribution:\n{df['Category'].value_counts()}")

# Clean text
print("Cleaning text...")
df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

# Handle empty texts
df = df[df['Cleaned_Resume'].str.len() > 50]

X = df['Cleaned_Resume']
y = df['Category']

# TF-IDF with improved parameters
print("Vectorizing text...")
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.85  # Ignore terms that appear in more than 85% of documents
)
X_tfidf = tfidf.fit_transform(X)

print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Model with hyperparameter tuning
print("Training model...")
clf = LinearSVC(
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    max_iter=2000
)

# Simple parameter grid for tuning (optional)
param_grid = {
    'C': [0.1, 1, 10]
}

# Uncomment for hyperparameter tuning (takes longer)
# clf = GridSearchCV(
#     LinearSVC(random_state=42, class_weight='balanced', max_iter=2000),
#     param_grid,
#     cv=3,
#     n_jobs=-1
# )

clf.fit(X_train, y_train)

# Evaluate
print("\n=== Model Evaluation ===")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
print("\nSaving model...")
import os
os.makedirs("model", exist_ok=True)

pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(clf, open("model/clf.pkl", "wb"))

# Save label mapping for reference
label_mapping = {i: label for i, label in enumerate(y.unique())}
pickle.dump(label_mapping, open("model/label_mapping.pkl", "wb"))

print("Model training completed and saved!")