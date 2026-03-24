import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('IMDB Dataset.csv')

print("First 5 rows:\n", df.head())

df = df.sample(50000, random_state=42)

reviews = df['review']
sentiments = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
acc_remove = accuracy_score(y_test, y_pred)
print(f"\nAccuracy with stopwords removed: {acc_remove:.4f}")
print(classification_report(y_test, y_pred))

vectorizer_no_stop = TfidfVectorizer(stop_words=None, max_features=5000)
X_train_tfidf_no_stop = vectorizer_no_stop.fit_transform(X_train)
X_test_tfidf_no_stop = vectorizer_no_stop.transform(X_test)

model_no_stop = LogisticRegression(max_iter=1000)
model_no_stop.fit(X_train_tfidf_no_stop, y_train)
y_pred_no_stop = model_no_stop.predict(X_test_tfidf_no_stop)
acc_keep = accuracy_score(y_test, y_pred_no_stop)
print(f"\nAccuracy with stopwords kept: {acc_keep:.4f}")

conditions = ['Stopwords Removed', 'Stopwords Kept']
accuracies = [acc_remove, acc_keep]

plt.bar(conditions, accuracies, color=['green', 'red'])
plt.ylabel('Accuracy')
plt.title('Effect of Stopword Removal on Sentiment Analysis')
plt.ylim(0, 1)
plt.show()