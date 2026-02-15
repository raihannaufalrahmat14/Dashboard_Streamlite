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

Klasifikasi:
• Positif  
• Netral  
• Negatif
""")

st.sidebar.markdown("---")

st.sidebar.header("⚙️ Metode")

st.sidebar.write("""
Algoritma : Support Vector Machine  
Feature Extraction : TF-IDF  
Framework : Streamlit
""")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📑 Menu Navigasi",
    [
        "Prediksi Sentimen",
        "Evaluasi Model",
        "Visualisasi Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("Sentiment Analysis v1.0")


# ====================================
# LOAD NLP
# ====================================

@st.cache_resource
def load_nlp():

    nltk.download("stopwords")

    stop_words = set(stopwords.words("indonesian"))

    factory = StemmerFactory()

    stemmer = factory.create_stemmer()

    return stop_words, stemmer


stop_words, stemmer = load_nlp()


# ====================================
# PREPROCESSING
# ====================================

def cleaning(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

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
Sistem ini mengklasifikasikan sentimen ulasan pengguna
aplikasi Grab menggunakan algoritma Support Vector Machine.
""")


# ====================================
# MENU 1: PREDIKSI
# ====================================

if menu == "Prediksi Sentimen":

    st.header("Prediksi Sentimen")

    text = st.text_area("Masukkan ulasan:")

    if st.button("Prediksi"):

        if text == "":
            st.warning("Masukkan teks terlebih dahulu")

        else:

            processed = preprocess(text)

            vector = vectorizer.transform([processed])

            result = model.predict(vector)[0]

            if result == "positif":
                st.success("Sentimen: POSITIF")

            elif result == "netral":
                st.info("Sentimen: NETRAL")

            else:
                st.error("Sentimen: NEGATIF")


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

    labels = ["negatif", "netral", "positif"]

    # accuracy
    acc = accuracy_score(y_true, y_pred)

    st.subheader("Akurasi")

    st.success(f"{acc:.4f}")


    # confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

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


    # classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0
    )

    st.text(report)


# ====================================
# MENU 3: VISUALISASI
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


    # ====================================
    # BAR PLOT
    # ====================================

    st.subheader("Distribusi Sentimen (Bar Plot)")

    fig, ax = plt.subplots()

    sns.barplot(
        x=summary.index,
        y=summary.values
    )

    st.pyplot(fig)


    # ====================================
    # DONUT PLOT
    # ====================================

    st.subheader("Distribusi Sentimen (Donut Plot)")

    fig, ax = plt.subplots()

    sizes = summary.values

    labels = summary.index

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )

    centre_circle = plt.Circle((0,0),0.70,fc="white")

    fig.gca().add_artist(centre_circle)

    ax.axis("equal")

    st.pyplot(fig)


    # ====================================
    # WORDCLOUD PER SENTIMEN
    # ====================================

    st.subheader("WordCloud per Sentimen")

    df["clean"] = df["content"].apply(preprocess)

    pos = " ".join(df[df["sentimen"]=="positif"]["clean"])

    neu = " ".join(df[df["sentimen"]=="netral"]["clean"])

    neg = " ".join(df[df["sentimen"]=="negatif"]["clean"])

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write("Positif")

        wc = WordCloud(width=400, height=300).generate(pos)

        fig, ax = plt.subplots()

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)


    with col2:

        st.write("Netral")

        wc = WordCloud(width=400, height=300).generate(neu)

        fig, ax = plt.subplots()

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)


    with col3:

        st.write("Negatif")

        wc = WordCloud(width=400, height=300).generate(neg)

        fig, ax = plt.subplots()

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)
