import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

file_path = '/content/train.tsv'

df = pd.read_csv(file_path, sep='\t')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words or word in ['not', 'very']]
    return ' '.join(tokens)

df['cleaned_reviews'] = df['review'].apply(clean_text)

df = df[df['cleaned_reviews'].str.strip() != '']

X = df['cleaned_reviews']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_tfidf_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf_smote, y_train_smote)

model = grid_search.best_estimator_

model.fit(X_train_tfidf_smote, y_train_smote)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_sentiment(review):
    cleaned_review = clean_text(review)
    review_tfidf = vectorizer.transform([cleaned_review])
    sentiment = model.predict(review_tfidf)
    return 'Positive' if sentiment == 1 else 'Negative'

user_review = input("Enter a review: ")
sentiment = predict_sentiment(user_review)
print(f"Sentiment of the review: {sentiment}")
