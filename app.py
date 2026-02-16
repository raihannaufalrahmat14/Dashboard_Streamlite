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
        "Analisis Sentimen",
        "Processing",
        "Evaluasi Model",
        "Visualisasi Dataset"
    )
)

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
# PREPROCESSING FUNCTIONS
# ====================================

def case_folding(text):
    return str(text).lower()

def remove_url(text):
    return re.sub(r'http\S+', '', text)

def remove_symbol(text):
    return re.sub(r'[^a-z\s]', '', text)

def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def tokenizing(text):
    return text.split()

def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess(text):

    text = case_folding(text)
    text = remove_url(text)
    text = remove_symbol(text)
    text = remove_whitespace(text)

    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)

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
# LOAD DATASET
# ====================================

@st.cache_data
def load_data():

    df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

    def label(score):

        if score <= 2:
            return "negatif"
        elif score == 3:
            return "netral"
        else:
            return "positif"

    df["sentimen"] = df["score"].apply(label)

    return df


df = load_data()


# ====================================
# TITLE
# ====================================

st.title("📊 Sistem Analisis Sentimen Ulasan Grab (SVM)")


# ====================================
# MENU 1: ANALISIS SENTIMEN
# ====================================

if menu == "Analisis Sentimen":

    st.header("Analisis Sentimen")

    text = st.text_area("Masukkan ulasan:")

    if st.button("Analisis"):

        if text == "":
            st.warning("Masukkan teks terlebih dahulu")

        else:

            processed = preprocess(text)

            vector = vectorizer.transform([processed])

            prediction = model.predict(vector)[0]

            prediction = str(prediction).lower()

            st.subheader("Hasil Preprocessing")
            st.info(processed)

            if prediction == "positif":
                st.success("Sentimen: POSITIF")

            elif prediction == "netral":
                st.info("Sentimen: NETRAL")

            else:
                st.error("Sentimen: NEGATIF")


# ====================================
# MENU 2: PROCESSING (SUDAH DIPERBAIKI)
# ====================================

elif menu == "Processing":

    st.header("Tahapan Text Processing")

    # tampilkan hanya kolom tertentu
    st.subheader("Dataset Awal")

    df_display = df[["content", "score", "sentimen"]].rename(columns={
        "content": "Ulasan",
        "score": "Score",
        "sentimen": "Sentimen"
    })

    st.dataframe(df_display)


    # pilih index
    index = st.number_input(
        "Pilih index data untuk melihat proses:",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )


    original_text = df.loc[index, "content"]

    st.markdown("---")
    st.subheader("1. Teks Asli")
    st.write(original_text)


    case = case_folding(original_text)
    st.subheader("2. Case Folding")
    st.write(case)


    no_url = remove_url(case)
    st.subheader("3. Remove URL")
    st.write(no_url)


    no_symbol = remove_symbol(no_url)
    st.subheader("4. Remove Symbol")
    st.write(no_symbol)


    clean_text = remove_whitespace(no_symbol)
    st.subheader("5. Remove Whitespace")
    st.write(clean_text)


    tokens = tokenizing(clean_text)
    st.subheader("6. Tokenizing")
    st.write(tokens)


    no_stopword = stopword_removal(tokens)
    st.subheader("7. Stopword Removal")
    st.write(no_stopword)


    stemmed = stemming(no_stopword)
    st.subheader("8. Stemming")
    st.write(stemmed)


    final_text = " ".join(stemmed)

    st.subheader("9. Final Preprocessing")
    st.success(final_text)


    st.subheader("10. TF-IDF Vector")

    vector = vectorizer.transform([final_text])

    tfidf_df = pd.DataFrame(
        vector.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    st.dataframe(tfidf_df)


# ====================================
# MENU 3: EVALUASI MODEL
# ====================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")

    df["clean"] = df["content"].apply(preprocess)

    X = vectorizer.transform(df["clean"])

    y_true = df["sentimen"]
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)

    st.subheader("Akurasi")
    st.success(f"{acc:.2f}")

    st.subheader("Confusion Matrix")

    labels_order = ["negatif", "netral", "positif"]

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

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
# MENU 4: VISUALISASI DATASET
# ====================================

elif menu == "Visualisasi Dataset":

    st.header("Visualisasi Dataset")

    summary = df["sentimen"].value_counts()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Distribusi Sentimen")

        fig, ax = plt.subplots()

        sns.barplot(
            x=summary.index,
            y=summary.values
        )

        st.pyplot(fig)


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


    st.subheader("WordCloud")

    df["clean"] = df["content"].apply(preprocess)

    col1, col2, col3 = st.columns(3)

    for sentimen, col in zip(
        ["positif", "netral", "negatif"],
        [col1, col2, col3]
    ):

        text = " ".join(
            df[df["sentimen"] == sentimen]["clean"]
        )

        if text.strip() != "":

            wc = WordCloud(
                width=400,
                height=300,
                background_color="white"
            ).generate(text)

            fig, ax = plt.subplots()

            ax.imshow(wc)
            ax.axis("off")

            col.subheader(sentimen.upper())
            col.pyplot(fig)
