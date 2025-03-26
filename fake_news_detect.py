import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fake_news.csv")

# Check if required columns exist
if not all(col in df.columns for col in ['title', 'label']):
    print("Error: CSV file must contain 'text' and 'label' columns.")
    exit()

# Convert labels: 'FAKE' -> 1, 'REAL' -> 0
df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})

# Prepare data
X = df['title']
y = df['label']

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")
print(f"Accuracy:{accuracy}")
