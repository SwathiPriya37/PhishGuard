# 🛡️ PhishGuard: BERT-Based Phishing Detection

PhishGuard is an NLP-powered phishing detection model built using **BERT**.  
It classifies text (e.g., emails, URLs, or messages) as **phishing** or **legitimate**.  

The model is trained and hosted on **Hugging Face**, while this repository contains the **training pipeline, preprocessing scripts, and evaluation code**.

---

## 🚀 Model
The fine-tuned model is available on Hugging Face:  
👉 [bert-phishing-detector](https://huggingface.co/Swathi37/bert-phishing-detector)

You can load it directly in Python:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Swathi37/bert-phishing-detector")
model = AutoModelForSequenceClassification.from_pretrained("Swathi37/bert-phishing-detector")

text = "Your account has been suspended. Click here to verify."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```
## 📦 Installation
```bash
git clone https://github.com/SwathiPriya37/PhishGuard.git
cd PhishGuard
pip install -r requirements.txt
```
## 🏋️ Training
To fine-tune BERT on your phishing dataset:

```bash
python src/train_bert.py
```
## 📊 Evaluation
Evaluate the trained model:

```bash
python src/evaluate_model.py
```
## 📂 Using Your Own Dataset
To train on your own dataset, prepare a CSV file with the following format:

csv
text,label
"Your account is locked. Verify now.",phishing
"Meeting is scheduled at 3 PM tomorrow.",legitimate
Then run:

bash
```
python src/train_bert.py --data data/your_dataset.csv
```
## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to add.

## 📜 License
This project is licensed under the MIT License.

## 👩‍💻 Author
Developed by Swathi Priya
Model: Swathi37/bert-phishing-detector

---
