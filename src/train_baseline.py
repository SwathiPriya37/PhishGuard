# src/train_baseline.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data

df = pd.read_csv("data/processed/processed_data.csv")
print("Data shape:", df.shape)

# 2. Preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " URL ", text)  # replace URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove special chars
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# Handle empty text
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df.loc[df["clean_text"].str.strip() == "", "clean_text"] = "emptyemail"

# Features
X_text = df["clean_text"]
X_features = df.drop(columns=["text", "label", "clean_text"], errors="ignore")
y = df["label"]

# 3. Debug Checks

print("X_text length:", len(X_text))
print("X_features length:", len(X_features))
print("y length:", len(y))
print("Unique labels:", y.unique())

# 4. Train/Test Split

# If y has only 1 class, remove stratify
if len(y.unique()) > 1:
    stratify_arg = y
else:
    stratify_arg = None
    print("⚠️ Warning: Only one class found, disabling stratify.")

X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

# 5. Vectorize Text

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

# 6. Train Baseline Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
