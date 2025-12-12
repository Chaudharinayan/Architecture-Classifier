# Architecture Classifier (Microservices vs REST API vs Serverless)

This project implements a **text classification machine learning model** that reads a **one-page technical document** (PDF or text) and predicts whether it describes:

- **Microservices architecture**
- **REST API design**
- **Serverless solution**

This was built as part of an assignment to demonstrate basic ML preprocessing, simple model training, and prediction on real-world technical architecture documents.

---

## 🚀 Features

### ✔ Machine Learning Model
- TF–IDF vectorizer  
- Logistic Regression classifier  
- 3-class architecture prediction  
- ~90 handcrafted training examples (included in dataset.csv)

### ✔ PDF Reader
- Extracts text from PDFs using PyPDF2  
- Classifies architecture from the first page  
- Works directly from CLI

### ✔ CLI Commands
Train model:
```bash
python src/train_and_classify.py --train
```
Predict using pdf:
```bash
python src/train_and_classify.py --predict-pdf "path/to/document.pdf"
```
Predict using direct text:
```bash
python src/train_and_classify.py --predict-text "your architecture description here"
```
📖 How It Works

1. dataset.csv contains 90 labeled examples:
  30 microservices
  30 REST API
  30 serverless

2. Model trains using:
  TF-IDF (bigrams + stop words removed)
  Logistic Regression
3. PDF text is extracted → fed into classifier → output probabilities.
   
## Requirements
```bash
pip install -r requirements.txt
```
