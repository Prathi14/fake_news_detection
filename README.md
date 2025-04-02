# Fake-News-Detection-System
Fake News Detection System
This project is a Fake News Detection App built using Streamlit and Logistic Regression. The model predicts whether a given news title is Real or Fake based on text analysis.

Features
Simple and user-friendly Streamlit UI for news classification
TF-IDF Vectorization for feature extraction
Logistic Regression Model trained on a labeled dataset
81% accuracy on test data

Tech Stack
Python
Pandas for data manipulation
Scikit-Learn for machine learning
Streamlit for web app UI
Pickle for model storage

File Structure
Fake-News-Detection
│──  app.py             # Streamlit app for fake news detection
│──  train_model.py     # Model training and saving script
│──  model.pkl          # Trained Logistic Regression model
│──  vectorizer.pkl     # TF-IDF vectorizer
│──  fake_news.csv      # Dataset used for training
│──  README.md          # Project documentation (this file)

Getting Started

1️⃣ Clone the repository
git clone https://github.com/PNBisaleri/fake-news-detection-system.git
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit app
streamlit run app.py

How It Works
Train Model: Run train_model.py to train and save the model.
Use App: Enter a news title in the text box and click "Check".
Get Prediction: The app will classify the news as Real (✅) or Fake (❌).

Model Performance
Accuracy: 81%
Algorithm Used: Logistic Regression
Vectorization: TF-IDF (5000 features, stop words removed)

Contact
For any queries or contributions, feel free to reach out:
Email: pnbisaleri@gmail.com
GitHub: PNBisaleri

Made with ❤️ by Pavithra N Bisaleri
