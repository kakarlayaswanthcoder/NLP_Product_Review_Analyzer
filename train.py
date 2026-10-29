import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from nlp_utils import preprocess_text


# Load dataset
df = pd.read_csv("data/product_reviews.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)



df["processed_review"] = df["review"].apply(
    lambda x: " ".join(preprocess_text(x))
)

print("\nProcessed Reviews:")
print(df[["review", "processed_review"]].head())



X = df["processed_review"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    df["processed_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)



vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)



accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



test_reviews = [
    "This product is amazing and I love it",
    "This product is terrible and completely useless",
    "This phone is okay and works normally"
]

test_processed = [
    " ".join(preprocess_text(review))
    for review in test_reviews
]

test_features = vectorizer.transform(test_processed)

predictions = model.predict(test_features)

print("\nTest Predictions:")

for review, prediction in zip(test_reviews, predictions):
    print(f"{review} --> {prediction}")
    

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")