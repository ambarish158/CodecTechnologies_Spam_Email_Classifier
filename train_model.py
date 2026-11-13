
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 1: Load the dataset
print("Loading dataset...")

data = pd.read_csv(
    "dataset/SMSSpamCollection.tsv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# STEP 2: Prepare input and output
X = data["message"]
y = data["label"]

# STEP 3: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# STEP 4: Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# STEP 5: Train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

# STEP 6: Make predictions
predictions = model.predict(X_test)

# STEP 7: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# STEP 8: Display classification report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, predictions))

# STEP 9: Display confusion matrix
print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, predictions))

# STEP 10: Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/spam_classifier.pkl")

print("\nModel saved successfully!")
print("Location: models/spam_classifier.pkl")
