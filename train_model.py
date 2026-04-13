import pandas as pd
import re
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("STARTING TRAINING...")

# Load dataset
data = pd.read_csv("emails.csv")
print("Dataset loaded:")
print(data.head())

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

data['cleaned'] = data['text'].apply(clean_text)

# Keyword feature
keywords = ["urgent", "verify", "click", "password", "account"]
data['keyword_count'] = data['cleaned'].apply(
    lambda x: sum(word in x for word in keywords)
)

# Features
X_text = data['cleaned']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X_text)

X = np.hstack((X_vec.toarray(), data[['keyword_count']].values))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

models = {"Random Forest": rf, "Logistic Regression": lr}

best_model = None
best_score = 0

# Evaluate
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = model

# Save model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nFINISHED TRAINING - Best model saved.")
