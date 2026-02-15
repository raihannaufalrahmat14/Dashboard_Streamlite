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

menu = st.sidebar.radio(
    "Menu Navigasi",
    (
        "Prediksi Sentimen",
        "Evaluasi Model",
        "Visualisasi Dataset"
    )
)

st.sidebar.markdown("---")
st.sidebar.info("Streamlit Sentiment Analysis v2.0")


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
# LOAD MODEL & VECTORIZER
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

st.title("📊 Sistem Analisis Sentimen Ulasan Grab (SVM)")


# ====================================
# MENU 1: PREDIKSI
# ====================================

if menu == "Prediksi Sentimen":

    st.header("Prediksi Sentimen")

    text_input = st.text_area("Masukkan ulasan:")

    if st.button("Prediksi"):

        if text_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            processed = preprocess(text_input)
            vector = vectorizer.transform([processed])
            prediction = model.predict(vector)[0]
            prediction = str(prediction).lower().strip()

            st.subheader("Hasil Prediksi")

            if prediction == "positif":
                st.success("Sentimen: POSITIF")

            elif prediction == "netral":
                st.info("Sentimen: NETRAL")

            elif prediction == "negatif":
                st.error("Sentimen: NEGATIF")

            else:
                st.warning(f"Label tidak dikenali: {prediction}")


# ====================================
# MENU 2: EVALUASI MODEL
# ====================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")

    df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

    # Labeling berdasarkan rating
    def label(score):
        if score <= 2:
            return "negatif"
        elif score == 3:
            return "netral"
        else:
            return "positif"

    df["sentimen"] = df["score"].apply(label)
    df["clean"] = df["content"].apply(preprocess)

    X = vectorizer.transform(df["clean"])
    y_true = df["sentimen"].astype(str).str.lower().str.strip()
    y_pred = pd.Series(model.predict(X)).astype(str).str.lower().str.strip()

    labels_order = ["negatif", "netral", "positif"]

    # ACCURACY
    acc = accuracy_score(y_true, y_pred)

    st.subheader("Akurasi")
    st.success(f"{acc:.4f}")

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_order
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order
    )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    st.pyplot(fig)

    # CLASSIFICATION REPORT
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=labels_order,
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

    col1, col2 = st.columns(2)

    # BAR CHART
    with col1:
        st.subheader("Distribusi Sentimen")

        fig, ax = plt.subplots()
        sns.barplot(x=summary.index, y=summary.values)
        st.pyplot(fig)

    # DONUT CHART
    with col2:
        st.subheader("Donut Chart")

        fig, ax = plt.subplots()

        ax.pie(
            summary.values,
            labels=summary.index,
            autopct="%1.1f%%",
            wedgeprops=dict(width=0.4)
        )

        ax.set_aspect("equal")
        st.pyplot(fig)

    # WORDCLOUD
    st.subheader("WordCloud per Sentimen")

    df["clean"] = df["content"].apply(preprocess)

    col1, col2, col3 = st.columns(3)

    for sentimen, col in zip(["positif", "netral", "negatif"], [col1, col2, col3]):

        text_data = " ".join(df[df["sentimen"] == sentimen]["clean"])

        if text_data.strip() != "":
            wc = WordCloud(
                width=400,
                height=300,
                background_color="white"
            ).generate(text_data)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")

            col.subheader(sentimen.upper())
            col.pyplot(fig)
