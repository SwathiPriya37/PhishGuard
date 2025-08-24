import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW   

# Load dataset
print("Loading dataset...")
df = pd.read_csv("phishing_clean.csv")   # make sure path is correct

print("Initial data shape:", df.shape)
print(df.head())

# Use the correct text column
df['text_combined'] = df['text_combined'].fillna("")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text_combined'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create datasets
train_dataset = PhishingDataset(X_train, y_train, tokenizer)
test_dataset = PhishingDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop (1 epoch for demo)
print("Starting training...")
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
print("Training complete!")

# Evaluation
print("Evaluating...")
model.eval()
preds, truths = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        preds.extend(predictions.cpu().numpy())
        truths.extend(labels.cpu().numpy())

print(classification_report(truths, preds))
