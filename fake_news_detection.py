import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

fake_data = pd.read_csv('D:/Python Program/AI Project/Fake.csv') 
true_data = pd.read_csv('D:/Python Program/AI Project/True.csv')

fake_data['label'] = 1  # Fake
true_data['label'] = 0  # Real

data = pd.concat([fake_data, true_data], ignore_index=True)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

data['cleaned_text'] = data['text'].apply(preprocess_text).apply(remove_stopwords)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['cleaned_text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

user_input = input("Enter the news article you want to check:\n")
user_input_cleaned = remove_stopwords(preprocess_text(user_input))
user_tfidf = tfidf.transform([user_input_cleaned])
prediction = model.predict(user_tfidf)

if prediction[0] == 1:
    print("\n\nThe article is predicted to be: FAKE")
else:
    print("\n\nThe article is predicted to be: REAL")
