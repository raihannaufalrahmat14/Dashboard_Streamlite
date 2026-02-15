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
# SIDEBAR
# ====================================

st.sidebar.title("📊 Dashboard Analisis Sentimen")

st.sidebar.markdown("---")

st.sidebar.header("📌 Tentang Sistem")

st.sidebar.write("""
Sistem ini digunakan untuk melakukan analisis sentimen
ulasan pengguna aplikasi Grab menggunakan algoritma
Support Vector Machine (SVM).

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
Framework : Streamlit  
Bahasa : Python  
""")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📑 Menu Navigasi",
    (
        "Prediksi Sentimen",
        "Evaluasi Model",
        "Visualisasi Dataset"
    )
)

st.sidebar.markdown("---")

st.sidebar.header("👨‍🎓 Pengembang")

st.sidebar.write("""
Nama : Raihan Kimo  
Penelitian : Analisis Sentimen Grab  
Metode : Support Vector Machine
""")

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
Sistem ini mengklasifikasikan sentimen ulasan pengguna aplikasi Grab
menggunakan algoritma Support Vector Machine (SVM).
""")


# ====================================
# LABEL FUNCTION
# ====================================

def label(score):

    if score <= 2:
        return "negatif"
    elif score == 3:
        return "netral"
    else:
        return "positif"


# ====================================
# MENU 1: PREDIKSI
# ====================================

if menu == "Prediksi Sentimen":

    st.header("Prediksi Sentimen")

    text = st.text_area("Masukkan ulasan pengguna:")

    if st.button("Prediksi"):

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

    y_true = df["score"].apply(label)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)

    st.subheader("Akurasi Model")

    st.success(f"Akurasi Model: {accuracy:.2%}")


    # Confusion Matrix BENAR

    st.subheader("Confusion Matrix")

    labels = ["negatif", "netral", "positif"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    ax.set_xlabel("Prediksi")

    ax.set_ylabel("Aktual")

    st.pyplot(fig)


    # Classification Report

    st.subheader("Classification Report")

    report = classification_report(y_true, y_pred, target_names=labels)

    st.text(report)


# ====================================
# MENU 3: VISUALISASI DATASET
# ====================================

elif menu == "Visualisasi Dataset":

    st.header("Visualisasi Dataset")

    df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

    df["sentimen"] = df["score"].apply(label)

    summary = df["sentimen"].value_counts()


    # Bar Chart

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


    # WORDCLOUD TERPISAH

    st.subheader("WordCloud Berdasarkan Sentimen")

    df["clean"] = df["content"].apply(preprocess)

    positif_text = " ".join(df[df["sentimen"]=="positif"]["clean"])
    netral_text = " ".join(df[df["sentimen"]=="netral"]["clean"])
    negatif_text = " ".join(df[df["sentimen"]=="negatif"]["clean"])

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write("Positif")

        if positif_text.strip() != "":
            wc = WordCloud(width=400, height=300, background_color="white").generate(positif_text)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)


    with col2:

        st.write("Netral")

        if netral_text.strip() != "":
            wc = WordCloud(width=400, height=300, background_color="white").generate(netral_text)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)


    with col3:

        st.write("Negatif")

        if negatif_text.strip() != "":
            wc = WordCloud(width=400, height=300, background_color="white").generate(negatif_text)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
