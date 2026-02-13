import streamlit as st
import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Analisis Sentimen Grab",
    layout="wide"
)

# =============================
# DOWNLOAD STOPWORDS ONLY
# =============================
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('indonesian'))

stop_words = load_stopwords()

# =============================
# STEMMER
# =============================
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

# =============================
# CLEANING
# =============================
def cleaning(text):

    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =============================
# PREPROCESS (NO NLTK TOKENIZER)
# =============================
def preprocess_text(text):

    text = cleaning(text)

    tokens = text.split()

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

        st.error("File model tidak ditemukan")

        st.stop()

model, vectorizer = load_model()

# =============================
# LOAD DATASET
# =============================
@st.cache_data
def load_data():

    try:

        df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

    except:

        st.error("File grab_reviews.csv tidak ditemukan")

        st.stop()

    df['clean_text'] = df['content'].apply(preprocess_text)

    def label(score):

        if score <= 2:

            return "negatif"

        elif score == 3:

            return "netral"

        else:

            return "positif"

    df['sentimen'] = df['score'].apply(label)

    summary = df['sentimen'].value_counts().reset_index()

    summary.columns = ["Sentimen", "Jumlah"]

    summary["Persentase"] = summary["Jumlah"] / summary["Jumlah"].sum() * 100

    positif = " ".join(df[df.sentimen=="positif"].clean_text)

    netral = " ".join(df[df.sentimen=="netral"].clean_text)

    negatif = " ".join(df[df.sentimen=="negatif"].clean_text)

    return summary, positif, netral, negatif

summary, positif_text, netral_text, negatif_text = load_data()

# =============================
# TITLE
# =============================
st.title("📊 Aplikasi Analisis Sentimen Ulasan Grab Menggunakan SVM")

st.write("Model Support Vector Machine digunakan untuk mengklasifikasikan sentimen ulasan pengguna aplikasi Grab.")

# =============================
# INPUT PREDIKSI
# =============================
st.header("Prediksi Sentimen")

user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):

    if user_input == "":

        st.warning("Masukkan teks terlebih dahulu")

    else:

        processed = preprocess_text(user_input)

        vector = vectorizer.transform([processed])

        hasil = model.predict(vector)[0]

        if hasil == "positif":

            st.success("Sentimen: POSITIF")

        elif hasil == "netral":

            st.info("Sentimen: NETRAL")

        else:

            st.error("Sentimen: NEGATIF")

# =============================
# VISUALISASI
# =============================
st.header("Visualisasi Distribusi Sentimen")

col1, col2 = st.columns(2)

# BAR CHART
with col1:

    fig, ax = plt.subplots()

    sns.barplot(x="Sentimen", y="Jumlah", data=summary, ax=ax)

    ax.set_title("Distribusi Sentimen")

    st.pyplot(fig)

# DONUT CHART
with col2:

    fig, ax = plt.subplots()

    ax.pie(summary["Persentase"], autopct="%1.1f%%")

    centre = plt.Circle((0,0),0.70,fc="white")

    fig.gca().add_artist(centre)

    ax.set_title("Persentase Sentimen")

    st.pyplot(fig)

# =============================
# WORDCLOUD
# =============================
st.header("WordCloud")

c1, c2, c3 = st.columns(3)

def tampil_wordcloud(text, title, col):

    with col:

        st.subheader(title)

        if text == "":

            st.write("Tidak ada data")

        else:

            wc = WordCloud(width=400, height=200, background_color="white").generate(text)

            fig, ax = plt.subplots()

            ax.imshow(wc)

            ax.axis("off")

            st.pyplot(fig)

tampil_wordcloud(positif_text, "Positif", c1)

tampil_wordcloud(netral_text, "Netral", c2)

tampil_wordcloud(negatif_text, "Negatif", c3)
