import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data_path = os.path.join("data", "dataset.txt")

texts = []
labels = []

with open(data_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(";")
        if len(parts) == 2:
            texts.append(parts[0])
            labels.append(parts[1])

# Convert to DataFrame
df = pd.DataFrame({"text": texts, "label": labels})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 🔥 Improved TF-IDF (key upgrade)
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 Stronger model
model = LinearSVC(C=1.0)

model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 Improved Model Accuracy: {accuracy * 100:.2f}%\n")

# Detailed report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")