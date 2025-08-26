import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
print("Loading dataset...")
df = pd.read_csv("phishing_clean.csv")
df['text_combined'] = df['text_combined'].fillna("")
texts = df['text_combined'].tolist()
labels = df['label'].tolist()

# Load tokenizer and model
print("Loading model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./saved_bert_model")  # path where you saved model
model.eval()

# Tokenize
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
loader = DataLoader(dataset, batch_size=16)

# Evaluate
print("Evaluating...")
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        input_ids, attention_mask, labels_batch = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels_batch.tolist())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))
