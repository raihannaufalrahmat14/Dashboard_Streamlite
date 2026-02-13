import streamlit as st
import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Dashboard Analisis Sentimen Grab",
    page_icon="📊",
    layout="wide"
)


# ====================================
# SIDEBAR PROFESIONAL
# ====================================

st.sidebar.title("📊 Dashboard Analisis Sentimen")

st.sidebar.markdown("---")

st.sidebar.header("📌 Tentang Sistem")

st.sidebar.write("""
Sistem ini digunakan untuk melakukan analisis sentimen
terhadap ulasan pengguna aplikasi Grab menggunakan
algoritma Support Vector Machine (SVM).

Klasifikasi sentimen:

• Positif  
• Netral  
• Negatif
""")

st.sidebar.markdown("---")

st.sidebar.header("⚙️ Metode")

st.sidebar.write("""
Algoritma : Support Vector Machine (SVM)  
Feature Extraction : TF-IDF  
Bahasa : Python  
Framework : Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.header("📑 Menu Navigasi")

menu = st.sidebar.radio(
    "Pilih Menu:",
    (
        "Prediksi Sentimen",
        "Evaluasi Model",
        "Visualisasi Dataset"
    )
)

st.sidebar.markdown("---")

st.sidebar.header("👨‍🎓 Informasi Pengembang")

st.sidebar.write("""
Nama : Raihan Kimo  
Penelitian : Analisis Sentimen Grab  
Metode : Support Vector Machine  
""")

st.sidebar.markdown("---")

st.sidebar.info("Streamlit Sentiment Analysis v1.0")


# ====================================
# LOAD NLP
# ====================================

@st.cache_resource
def load_nlp():

    nltk.download('stopwords')

    stop_words = set(stopwords.words('indonesian'))

    factory = StemmerFactory()

    stemmer = factory.create_stemmer()

    return stop_words, stemmer


stop_words, stemmer = load_nlp()


# ====================================
# PREPROCESSING
# ====================================

def cleaning(text):

    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess(text):

    text = cleaning(text)

    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


# ====================================
# LOAD MODEL
# ====================================

@st.cache_resource
def load_model():

    model = joblib.load("best_svm_model.pkl")

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# ====================================
# TITLE
# ====================================

st.title("📊 Sistem Analisis Sentimen Ulasan Grab")

st.write("""
Aplikasi ini mengklasifikasikan sentimen ulasan pengguna
aplikasi Grab menggunakan algoritma Support Vector Machine (SVM).
""")


# ====================================
# MENU 1: PREDIKSI
# ====================================

if menu == "Prediksi Sentimen":

    st.header("Prediksi Sentimen")

    text = st.text_area("Masukkan ulasan pengguna:")

    if st.button("Prediksi Sentimen"):

        if text == "":

            st.warning("Masukkan teks terlebih dahulu")

        else:

            processed = preprocess(text)

            vector = vectorizer.transform([processed])

            prediction = model.predict(vector)[0]

            if prediction == "positif":

                st.success("Hasil Prediksi: POSITIF")

            elif prediction == "netral":

                st.info("Hasil Prediksi: NETRAL")

            else:

                st.error("Hasil Prediksi: NEGATIF")


# ====================================
# MENU 2: EVALUASI MODEL
# ====================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")

    df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

    df["clean"] = df["content"].apply(preprocess)

    X = vectorizer.transform(df["clean"])

    def label(score):

        if score <= 2:
            return "negatif"

        elif score == 3:
            return "netral"

        else:
            return "positif"

    y_true = df["score"].apply(label)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)

    st.subheader("Akurasi Model")

    st.success(f"Akurasi: {accuracy:.2f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negatif", "netral", "positif"],
        yticklabels=["negatif", "netral", "positif"]
    )

    ax.set_xlabel("Prediksi")

    ax.set_ylabel("Aktual")

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(y_true, y_pred)

    st.text(report)


# ====================================
# MENU 3: VISUALISASI DATASET
# ====================================

elif menu == "Visualisasi Dataset":

    st.header("Visualisasi Dataset")

    df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

    def label(score):

        if score <= 2:
            return "negatif"

        elif score == 3:
            return "netral"

        else:
            return "positif"

    df["sentimen"] = df["score"].apply(label)

    summary = df["sentimen"].value_counts()

    st.subheader("Distribusi Sentimen")

    fig, ax = plt.subplots()

    sns.barplot(
        x=summary.index,
        y=summary.values,
        palette="viridis"
    )

    ax.set_xlabel("Sentimen")

    ax.set_ylabel("Jumlah")

    st.pyplot(fig)

    st.subheader("WordCloud")

    df["clean"] = df["content"].apply(preprocess)

    text = " ".join(df["clean"])

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots()

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)
