
import os
import argparse
import pickle
import zipfile

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

ARTIFACT_DIR = "artifacts"
MODEL_FILE = os.path.join(ARTIFACT_DIR, "arch_model.pkl")
VECTORIZER_FILE = os.path.join(ARTIFACT_DIR, "arch_vectorizer.pkl")
ZIP_FILE = os.path.join(ARTIFACT_DIR, "architecture_model_bundle.zip")

def train_and_save(data_path="dataset.csv", out_dir=ARTIFACT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("dataset.csv must contain 'text' and 'label' columns")

    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print("\n=== Evaluation on test set ===")
    print(classification_report(y_test, y_pred, digits=4))
    labels_order = sorted(list(set(labels)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_order], columns=[f"pred_{l}" for l in labels_order])
    print("\nConfusion matrix:")
    print(cm_df)

    model_bundle = {'vectorizer': vectorizer, 'classifier': clf, 'labels': labels_order}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_bundle, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    with zipfile.ZipFile(ZIP_FILE, "w") as z:
        z.write(MODEL_FILE, arcname=os.path.basename(MODEL_FILE))
        z.write(VECTORIZER_FILE, arcname=os.path.basename(VECTORIZER_FILE))

    print(f"\nArtifacts saved to {out_dir}/")
    print(f"- {MODEL_FILE}")
    print(f"- {VECTORIZER_FILE}")
    print(f"- {ZIP_FILE}")
    return model_bundle

def load_model(bundle_path=MODEL_FILE):
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"{bundle_path} not found — train first with --train")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def extract_text_from_pdf(pdf_path, page_to_use=0):
    
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is not installed. Install with `pip install PyPDF2`.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    text_pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        if page_to_use < num_pages:
            try:
                page = reader.pages[page_to_use]
                page_text = page.extract_text() or ""
                return page_text.strip()
            except Exception:
                pass
        for p in reader.pages:
            page_text = p.extract_text() or ""
            text_pages.append(page_text)
    return "\n".join(text_pages).strip()

def classify_text(text, bundle):
    
    vectorizer = bundle['vectorizer']
    clf = bundle['classifier']
    labels = bundle['labels']
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]
    probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return pred, probs_dict

def cli():
    parser = argparse.ArgumentParser(description="Train model and classify PDFs/text for architecture types.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train model from dataset.csv and save artifacts.")
    group.add_argument("--predict-pdf", type=str, help="Path to PDF to classify (reads first page).")
    group.add_argument("--predict-text", type=str, help="Raw text string to classify (pass the doc as a string).")
    parser.add_argument("--use-page", type=int, default=0, help="PDF page index to read (default=0, first page).")
    args = parser.parse_args()

    if args.train:
        train_and_save()
        return

    bundle = load_model()

    if args.predict_pdf:
        pdf_path = args.predict_pdf
        print(f"Reading PDF (page {args.use_page}): {pdf_path}")
        try:
            text = extract_text_from_pdf(pdf_path, page_to_use=args.use_page)
        except Exception as e:
            print("Error reading PDF:", e)
            return
        if not text.strip():
            print("No text extracted from PDF. Try another page index or ensure PDF contains selectable text (not scanned images).")
            return
        pred, probs = classify_text(text, bundle)
        print("\n--- Prediction result ---")
        print("Predicted label:", pred)
        print("Probabilities:")
        for k,v in sorted(probs.items(), key=lambda x:-x[1]):
            print(f" - {k}: {v:.4f}")
        print("\n(Showing top labels. Provide one-page PDFs for most reliable results.)")
        return

    if args.predict_text:
        text = args.predict_text
        pred, probs = classify_text(text, bundle)
        print("\n--- Prediction result ---")
        print("Predicted label:", pred)
        print("Probabilities:")
        for k,v in sorted(probs.items(), key=lambda x:-x[1]):
            print(f" - {k}: {v:.4f}")
        return

if __name__ == "__main__":
    cli()