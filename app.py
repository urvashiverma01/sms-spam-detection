from flask import Flask, request, render_template_string
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Train model on startup
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'SMSSpamCollection'), sep='\t', header=None, names=['label', 'message'])

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
model = MultinomialNB()
model.fit(X, df['label'])

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>SMS Spam Detector</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 60px auto; background: #f4f4f4; }
        h1 { color: #6B2D8B; }
        textarea { width: 100%; padding: 10px; font-size: 16px; height: 120px; }
        button { background: #6B2D8B; color: white; padding: 10px 30px; border: none; font-size: 16px; cursor: pointer; margin-top: 10px; }
        .result { margin-top: 20px; padding: 20px; font-size: 22px; font-weight: bold; border-radius: 8px; }
        .spam { background: #ffe0e0; color: red; }
        .ham { background: #e0ffe0; color: green; }
    </style>
</head>
<body>
    <h1>📱 SMS Spam Detector</h1>
    <p>Enter any SMS message to check if it is Spam or Ham</p>
    <form method="POST">
        <textarea name="message" placeholder="Type your message here...">{{ message }}</textarea><br>
        <button type="submit">Check Message</button>
    </form>
    {% if result %}
    <div class="result {{ result_class }}">
        {{ result }}
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    result_class = None
    message = ''
    if request.method == 'POST':
        message = request.form['message']
        cleaned = clean_text(message)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == 'spam':
            result = '🚨 SPAM — This message looks suspicious!'
            result_class = 'spam'
        else:
            result = '✅ HAM — This message looks safe!'
            result_class = 'ham'
    return render_template_string(HTML, result=result, result_class=result_class, message=message)

if __name__ == '__main__':
    app.run(debug=True)