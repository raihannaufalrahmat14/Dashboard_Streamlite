import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. NLTK Data Downloads (Mencegah LookupError) ---
def download_nltk_resources():
    resources = {
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/stopwords': 'stopwords'
    }
    for path, resource in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# --- 2. Global Initializations ---
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- 3. Preprocessing Functions ---
def cleaning(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    cleaned_text = cleaning(text)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# --- 4. Load Model and TF-IDF Vectorizer ---
@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load('best_svm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: File model (.pkl) tidak ditemukan.")
        st.stop()

best_svm_model, tfidf_vectorizer = load_ml_models()

# --- 5. Load Data & Evaluation (DEBUGGING) ---
@st.cache_data
def load_and_evaluate():
    try:
        df = pd.read_csv('grab_reviews.csv', sep=';', encoding='latin1')
    except FileNotFoundError:
        st.error("Error: 'grab_reviews.csv' tidak ditemukan.")
        st.stop()

    # Preprocessing DataFrame
    df['final_text'] = df['content'].apply(cleaning).apply(preprocess_text)
    
    def label_sentiment(score):
        if score <= 2: return 'negatif'
        elif score == 3: return 'netral'
        else: return 'positif'
    
    df['sentimen'] = df['score'].apply(label_sentiment)

    # --- DEBUGGING: Cek Data Sebelum Split ---
    st.write("--- Debugging: Pengecekan Data ---")
    st.write("Contoh data setelah preprocessing (kolom 'final_text'):")
    st.write(df[['content', 'final_text', 'sentimen']].head())
    st.write("Jumlah data kosong setelah preprocessing:", df[df['final_text'] == '']['final_text'].count())
    st.write("Distribusi Sentimen Asli:", df['sentimen'].value_counts())
    st.write("-----------------------------------")
    # ----------------------------------------

    # Split Data & Hitung Confusion Matrix
    X = df['final_text']
    y = df['sentimen']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- DEBUGGING: Cek Data Uji ---
    st.write("Jumlah data uji:", len(y_test))
    # -------------------------------

    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = best_svm_model.predict(X_test_tfidf)
    
    # Cek label unik yang diprediksi
    st.write("Label yang diprediksi model:", pd.Series(y_pred).unique())

    cm = confusion_matrix(y_test, y_pred, labels=['negatif', 'netral', 'positif'])
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Siapkan Teks Wordcloud
    pos = " ".join(df[df['sentimen'] == 'positif']['final_text'])
    neu = " ".join(df[df['sentimen'] == 'netral']['final_text'])
    neg = " ".join(df[df['sentimen'] == 'negatif']['final_text'])

    return temp_df, pos, neu, neg, cm, report

# Panggil fungsi debugging
temp_df, positive_text, neutral_text, negative_text, cm, report = load_and_evaluate()

# --- 6. Streamlit UI (Sama seperti sebelumnya) ---
st.set_page_config(page_title="Grab Sentiment Analysis", layout="wide")
st.title("Aplikasi Analisis Sentimen Ulasan Grab")

# ... (Lanjutkan dengan kode UI seperti sebelumnya) ...
# ... (Prediksi Tunggal, Distribusi, Confusion Matrix, WordCloud) ...
