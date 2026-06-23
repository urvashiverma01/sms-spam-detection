import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# ── 2. CLEAN TEXT ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)

# ── 3. CONVERT TEXT TO NUMBERS (TF-IDF) ───────────────────
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# ── 4. SPLIT DATA ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. TRAIN MODEL ────────────────────────────────────────
model = MultinomialNB()
model.fit(X_train, y_train)

# ── 6. TEST MODEL ─────────────────────────────────────────
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
# ── 7. TEST WITH YOUR OWN MESSAGE ─────────────────────────
print("\n--- Test Your Own Message ---")
while True:
    user_input = input("Enter a message (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)
    print(f"Result: This message is --> {prediction[0].upper()}\n")