import pandas as pd
import re
import os
from bs4 import BeautifulSoup

# 1. Load dataset
df = pd.read_csv("data/phishing_email.csv")
print("Original data:", df.shape)

# 2. Fill missing values (handle missing columns safely)
if 'subject' not in df.columns:
    df['subject'] = ""
else:
    df['subject'] = df['subject'].fillna("")

if 'body' not in df.columns:
    df['body'] = ""
else:
    df['body'] = df['body'].fillna("")

if 'from' not in df.columns:
    df['from'] = "unknown"
else:
    df['from'] = df['from'].fillna("unknown")

# 3. Combine subject + body
df['text'] = df['subject'] + " " + df['body']

# 4. Remove HTML tags
def clean_html(raw_html):
    return BeautifulSoup(raw_html, "lxml").get_text()

df['text'] = df['text'].apply(clean_html)

# 5. Clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

# 6. Extract sender domain
def extract_domain(email):
    if "@" in email:
        return email.split("@")[-1].lower()
    return "unknown"

df['sender_domain'] = df['from'].apply(extract_domain)

# 7. URL features (if urls column exists)
if 'urls' in df.columns:
    def url_features(urls):
        if pd.isna(urls) or urls == "":
            return 0, 0
        url_list = urls.split()
        return len(url_list), sum(1 for u in url_list if len(u) > 50)

    df[['num_urls', 'long_urls']] = df['urls'].apply(
        lambda x: pd.Series(url_features(x))
    )
else:
    df['num_urls'], df['long_urls'] = 0, 0

# 8. Final dataset
final_df = df[['clean_text', 'sender_domain', 'num_urls', 'long_urls', 'label']]
print(final_df.head())

# 9. Save cleaned dataset into data/cleaned/
os.makedirs("data/cleaned", exist_ok=True)
final_df.to_csv("data/cleaned/cleaned_emails.csv", index=False)
print("✅ Cleaned dataset saved to data/cleaned/cleaned_emails.csv")
