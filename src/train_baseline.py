# src/train_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

# 1. Load dataset
df = pd.read_csv("data/cleaned/cleaned_emails.csv")
print("Data shape:", df.shape)
print(df.head())

# 2. Handle missing clean_text
df['clean_text'] = df['clean_text'].fillna("")

# Fallback: if still empty, use sender_domain as text
df.loc[df['clean_text'].str.strip() == "", 'clean_text'] = df['sender_domain']

print("After fixing NaNs:", df.shape)
print(df[['clean_text', 'sender_domain']].head(10))

# 3. Features and labels
X_text = df['clean_text'].astype(str)
X_features = df[['num_urls', 'long_urls']]

y = df['label']
if y.dtype == object:
    y = LabelEncoder().fit_transform(y)

# 4. Train-test split
X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

# 6. Combine text + numeric features
X_train = sp.hstack([X_text_train_vec, X_feat_train])
X_test = sp.hstack([X_text_test_vec, X_feat_test])

# 7. Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
