import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW

# ======================
# 1. Load dataset
# ======================
print("Loading dataset...")
df = pd.read_csv("C:/Users/Swathi priya/OneDrive/Documents/PhishGuard/data/phishing_email.csv")   
print("Initial data shape:", df.shape)

# Use text_combined instead of clean_text
df['text'] = df['text_combined'].fillna("")

# ======================
# 2. Torch Dataset Class
# ======================
class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ======================
# 3. Tokenizer & Dataset
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

texts = df['text'].tolist()
labels = df['label'].tolist()

dataset = PhishingDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ======================
# 4. Model Setup
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Scheduler for learning rate
num_training_steps = len(dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ======================
# 5. Training Loop
# ======================
epochs = 3
model.train()

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

# ======================
# 6. Save Model
# ======================
model.save_pretrained("models/bert_phishing")
tokenizer.save_pretrained("models/bert_phishing")
print("\n✅ Model training complete. Saved at models/bert_phishing")
