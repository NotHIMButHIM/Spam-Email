#Email detection using Naive bayes
import warnings; warnings.filterwarnings('ignore')
import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline
import joblib

#load the dataset
file = os.path.join(os.path.dirname(__file__), "spam.csv")
df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')

#cleaning & preprocessing the data
df = df.dropna(subset=['label', 'text'])
df['label'] = df['label'].astype(int)
df['text'] = df['text'].str.lower().str.strip()
df = df[df['text'].str.len() > 0]

#train/test the data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label']) 

# Build & train pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
pipeline.fit(x_train, y_train)

# Evaluate
print(f"Accuracy: {pipeline.score(x_test, y_test) * 100:.2f}%")

# Save the pipeline
joblib.dump(pipeline, 'spam_pipeline.pkl')
