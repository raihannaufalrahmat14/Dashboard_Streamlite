import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Analisis Sentimen Grab",
    layout="wide"
)

# =============================
# DOWNLOAD NLTK DATA (SAFE FOR STREAMLIT CLOUD)
# =============================
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk()

# =============================
# LOAD STOPWORDS AND STEMMER
# =============================
@st.cache_resource
def load_nlp_tools():
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stop_words, stemmer

stop_words, stemmer = load_nlp_tools()

# =============================
# TEXT PREPROCESSING
# =============================
def cleaning(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = cleaning(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except:
        st.error("Model tidak ditemukan!")
        st.stop()

model, vectorizer = load_model()

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("grab_reviews.csv", sep=';', encoding='latin1')
    except:
        st.error("Dataset grab_reviews.csv tidak ditemukan!")
        st.stop()

    df['clean'] = df['content'].apply(preprocess_text)

    def label(score):
        if score <= 2:
            return "negatif"
        elif score == 3:
            return "netral"
        else:
            return "positif"

    df['sentimen'] = df['score'].apply(label)

    summary = df['sentimen'].value_counts().reset_index()
    summary.columns = ['Sentimen', 'Jumlah']
    summary['Persentase'] = summary['Jumlah'] / summary['Jumlah'].sum() * 100

    pos = " ".join(df[df.sentimen=="positif"].clean)
    neu = " ".join(df[df.sentimen=="netral"].clean)
    neg = " ".join(df[df.sentimen=="negatif"].clean)

    return summary, pos, neu, neg

summary, pos_text, neu_text, neg_text = load_data()

# =============================
# TITLE
# =============================
st.title("📊 Aplikasi Analisis Sentimen Ulasan Grab")

# =============================
# PREDICTION SECTION
# =============================
st.header("Prediksi Sentimen")

input_text = st.text_area(
    "Masukkan ulasan:",
    "Aplikasi ini sangat membantu"
)

if st.button("Prediksi"):

    if input_text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:

        processed = preprocess_text(input_text)
        vector = vectorizer.transform([processed])
        result = model.predict(vector)[0]

        if result == "positif":
            st.success("Hasil Prediksi: POSITIF 😊")

        elif result == "netral":
            st.info("Hasil Prediksi: NETRAL 😐")

        else:
            st.error("Hasil Prediksi: NEGATIF 😠")

# =============================
# BAR CHART
# =============================
st.header("Distribusi Sentimen")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots()

    sns.barplot(
        x="Sentimen",
        y="Jumlah",
        data=summary,
        ax=ax
    )

    st.pyplot(fig)

# =============================
# DONUT CHART
# =============================
with col2:

    fig, ax = plt.subplots()

    wedges, texts, autotexts = ax.pie(
        summary['Persentase'],
        autopct="%1.1f%%",
        startangle=90
    )

    centre = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre)

    ax.axis('equal')

    st.pyplot(fig)

# =============================
# WORDCLOUD
# =============================
st.header("WordCloud")

c1, c2, c3 = st.columns(3)

def show_wordcloud(text, title, column):

    with column:

        st.subheader(title)

        if text.strip() == "":
            st.write("Tidak ada data")
            return

        wc = WordCloud(
            width=400,
            height=200,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")

        st.pyplot(fig)

show_wordcloud(pos_text, "Positif", c1)
show_wordcloud(neu_text, "Netral", c2)
show_wordcloud(neg_text, "Negatif", c3)
