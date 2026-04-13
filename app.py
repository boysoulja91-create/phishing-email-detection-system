from flask import Flask, request, render_template
import pickle
import re
import numpy as np

app = Flask(__name__)

print("LOADING MODEL...")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

keywords = ["urgent", "verify", "click", "password", "account"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    cleaned = clean_text(email)

    vec = vectorizer.transform([cleaned]).toarray()
    keyword_count = sum(word in cleaned for word in keywords)

    final_input = np.hstack((vec, [[keyword_count]]))

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    if prediction == 1:
        result = f"⚠️ Phishing Email ({probability:.2f} confidence)"
    else:
        result = f"✅ Safe Email ({1 - probability:.2f} confidence)"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    print("STARTING FLASK APP...")
    app.run(debug=True)
