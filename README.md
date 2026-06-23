# SMS Spam Detection using NLP

A machine learning project that classifies SMS messages as Spam or Ham using NLP techniques.

## 🎯 Accuracy: 97%

## Tech Stack
- Python
- NLTK
- Scikit-learn (TF-IDF + Naive Bayes)
- Pandas

## How it works
1. Loads SMS Spam Collection dataset (5572 messages)
2. Cleans text (lowercase, remove punctuation, remove stopwords)
3. Converts text to numbers using TF-IDF Vectorization
4. Trains Multinomial Naive Bayes classifier
5. Predicts spam/ham on custom input in real time

## Result
| Class | Precision | Recall |
|-------|-----------|--------|
| Ham   | 0.97      | 1.00   |
| Spam  | 1.00      | 0.78   |
