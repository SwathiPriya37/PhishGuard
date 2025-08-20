# src/clean_data.py
import pandas as pd
import re
from bs4 import BeautifulSoup

# 1. Load dataset
df = pd.read_csv("data/phishing_email.csv")
print("Original data:", df.shape)
print("Columns in dataset:", df.columns.tolist())

# 2. Fill missing values safely
if "subject" in df.columns:
    df["subject"] = df["subject"].fillna("")
else:
    df["subject"] = ""  

if "body" in df.columns:
    df["body"] = df["body"].fillna("")
elif "text" in df.columns:
    df["body"] = df["text"].fillna("")
else:
    df["body"] = ""  # fallback

if "from" in df.columns:
    df["from"] = df["from"].fillna("unknown")
else:
    df["from"] = "unknown"

# 3. Combine subject + body
df["text"] = df["subject"] + " " + df["body"]

# 4. Remove HTML tags
def clean_html(raw_html):
    return BeautifulSoup(str(raw_html), "lxml").get_text()

df["text"] = df["text"].apply(clean_html)

# 5. Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"\S+@\S+", "", text)  # remove emails
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text.lower()

df["clean_text"] = df["text"].apply(clean_text)

# 6. Extract sender domain
def extract_domain(email):
    if "@" in str(email):
        return str(email).split("@")[-1].lower()
    return "unknown"

df["sender_domain"] = df["from"].apply(extract_domain)

# 7. URL features (if urls column exists)
if "urls" in df.columns:
    def url_features(urls):
        if pd.isna(urls) or urls == "":
            return 0, 0
        url_list = str(urls).split()
        return len(url_list), sum(1 for u in url_list if len(u) > 50)

    df[["num_urls", "long_urls"]] = df["urls"].apply(
        lambda x: pd.Series(url_features(x))
    )
else:
    df["num_urls"], df["long_urls"] = 0, 0

# 8. Final dataset
if "label" not in df.columns:
    raise ValueError("⚠️ Your dataset has no 'label' column!")

final_df = df[["clean_text", "sender_domain", "num_urls", "long_urls", "label"]]
print(final_df.head())

# 9. Save cleaned dataset
final_df.to_csv("data/cleaned_emails.csv", index=False)
print("✅ Cleaned dataset saved to data/cleaned_emails.csv")
