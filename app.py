import string
import pickle
import streamlit as st
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    filtered_words = []
    for word in text:
        if word.isalnum():
            filtered_words.append(word)

    text = filtered_words[:]
    filtered_words.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            filtered_words.append(word)

    text = filtered_words[:]
    filtered_words.clear()

    for word in text:
        filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)

# Load your dataset (e.g., CSV file)
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Data preprocessing
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.drop_duplicates(keep='first', inplace=True)

# Create 'transform_text' column
df['transform_text'] = df['text'].apply(transform_text)

# Training code
# Load or create your training data and labels
X_train = df['transform_text'].values
y_train = df['target'].values

# Create and fit a TF-IDF vectorizer to transform your text data into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=2)

# Create and fit a MultinomialNB model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Save the fitted vectorizer and model using pickle
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(nb_classifier, model_file)

# Streamlit application code
st.title("Email/SMS Spam Classifier")

input_message = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_message = transform_text(input_message)
    # Vectorize
    vector_input = tfidf_vectorizer.transform([transformed_message])
    # Predict
    result = nb_classifier.predict(vector_input)[0]
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
