from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# === Model Setup ===
data = pd.read_csv("phishing_emails.csv")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['email_text'])
y = data['label']
model = MultinomialNB()
model.fit(X, y)

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    confidence = model.predict_proba(email_vector)[0][prediction]

    result = (
        f"⚠️ PHISHING DETECTED! (Confidence: {confidence:.2f})"
        if prediction == 1 else
        f"✅ This email looks safe. (Confidence: {confidence:.2f})"
    )

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)