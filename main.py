import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("phishing_emails.csv")

# Show first few rows
print(df.head())

# Step 1: Set up the vectorizer (removes common words like "the", "is")
vectorizer = TfidfVectorizer(stop_words='english')

# Step 2: Convert email text to numbers
X = vectorizer.fit_transform(df['email_text'])

# Step 3: Get the label (0 = legit, 1 = phishing)
y = df['label']

# Step 4: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Create the model
model = MultinomialNB()

# Step 2: Train it on your training data
model.fit(X_train, y_train)

# Step 3: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 4: See how well it did
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


def test_email(email_text):
    # Vectorize the new input
    email_vector = vectorizer.transform([email_text])
    # Predict using the trained model
    prediction = model.predict(email_vector)[0]

    # Show result
    if prediction == 1:
        print("\n⚠️  PHISHING detected!")
    else:
        print("\n✅ This email looks safe.")


# Example test
test_email("Please update your billing information immediately to avoid suspension.")