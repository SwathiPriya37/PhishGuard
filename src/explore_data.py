# src/explore_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned/cleaned_emails.csv")
print("Data shape:", df.shape)
print(df.head())

# 1. Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", hue="label", palette="Set2", legend=False)
plt.title("Label Distribution (Ham vs Phish)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# 2. Text length distribution
df["text_length"] = df["clean_text"].fillna("").astype(str).apply(len)

plt.figure(figsize=(6, 4))
sns.histplot(df["text_length"], bins=50, kde=True)
plt.title("Distribution of Email Text Lengths")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.show()

# 3. Top sender domains
top_domains = df["sender_domain"].value_counts().head(10)

plt.figure(figsize=(8, 4))
sns.barplot(x=top_domains.index, y=top_domains.values, palette="viridis")
plt.title("Top 10 Sender Domains")
plt.xlabel("Domain")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 4. URL features
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="label", y="num_urls", palette="Set1")
plt.title("URLs per Email by Label")
plt.show()

print("✅ Data exploration completed!")
